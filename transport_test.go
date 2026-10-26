package main

// Advertising br and zstd is only honest if the router can read the answers. Go's
// transport decompresses gzip and nothing else, and only when it set the header —
// so an upstream that honours our full Accept-Encoding would hand back a body that
// fails to parse, and the failure would look like a provider fault.

import (
	"bytes"
	"compress/flate"
	"compress/gzip"
	"compress/zlib"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/andybalholm/brotli"
	"github.com/klauspost/compress/zstd"
)

func compressWith(t *testing.T, encoding string, payload []byte) []byte {
	t.Helper()
	var buf bytes.Buffer
	switch encoding {
	case "gzip":
		w := gzip.NewWriter(&buf)
		w.Write(payload)
		w.Close()
	case "deflate-zlib":
		w := zlib.NewWriter(&buf)
		w.Write(payload)
		w.Close()
	case "deflate-raw":
		w, err := flate.NewWriter(&buf, flate.DefaultCompression)
		if err != nil {
			t.Fatal(err)
		}
		w.Write(payload)
		w.Close()
	case "br":
		w := brotli.NewWriter(&buf)
		w.Write(payload)
		w.Close()
	case "identity":
		return payload
	case "zstd":
		w, err := zstd.NewWriter(&buf)
		if err != nil {
			t.Fatal(err)
		}
		w.Write(payload)
		w.Close()
	default:
		t.Fatalf("unknown encoding %q", encoding)
	}
	return buf.Bytes()
}

// encodingUpstream answers every request with the compression it was told to use and
// records the Accept-Encoding it saw.
func encodingUpstream(t *testing.T, encoding string, payload string) (*httptest.Server, func() (accepted string, calls int)) {
	t.Helper()
	var mu sync.Mutex
	var accepted string
	var calls int
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		mu.Lock()
		accepted, calls = r.Header.Get("Accept-Encoding"), calls+1
		mu.Unlock()

		headerEncoding := encoding
		switch encoding {
		case "deflate-zlib", "deflate-raw":
			headerEncoding = "deflate"
		case "identity":
			headerEncoding = ""
		}
		body := compressWith(t, encoding, []byte(payload))
		if headerEncoding != "" {
			w.Header().Set("Content-Encoding", headerEncoding)
		}
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(200)
		w.Write(body)
	}))
	t.Cleanup(srv.Close)
	return srv, func() (string, int) {
		mu.Lock()
		defer mu.Unlock()
		return accepted, calls
	}
}

func encodingRouter(t *testing.T, up *httptest.Server) *httptest.Server {
	t.Helper()
	orig := provs
	t.Cleanup(func() { provs = orig })
	provs = map[string]*Provider{
		"go": {Base: up.URL + "/go/v1", KeyEnv: "OPENCODE_API_KEY", SessionHeader: true,
			Models: map[string]ModelRoute{"mock-model": {ID: "mock-go"}}},
	}
	t.Setenv("OPENCODE_API_KEY", "k")
	led = NewLedger(t.TempDir() + "/spend.json")
	breakers = map[string]*breakerEntry{}
	shadow = false
	authToken = ""
	nowFn = func() time.Time { return mon(13) }
	srv := httptest.NewServer(newMux())
	t.Cleanup(srv.Close)
	return srv
}

// The provider must see the same compression set the mimicked client sends.
func TestAcceptEncodingMatchesClient(t *testing.T) {
	up, seen := encodingUpstream(t, "gzip",
		`{"choices":[{"message":{"content":"hi"},"finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":1}}`)
	router := encodingRouter(t, up)

	resp, raw := postAnthropic(t, router.URL, "/v1/messages",
		`{"model":"mock-model","max_tokens":16,"messages":[{"role":"user","content":"hi"}]}`)
	if resp.StatusCode != 200 {
		t.Fatalf("status = %d: %s", resp.StatusCode, raw)
	}
	accepted, _ := seen()
	// The literal the client was measured sending: comparing against the const
	// the code itself uses would pass no matter what the const held.
	want := "gzip, deflate, br, zstd"
	if accepted != want {
		t.Errorf("Accept-Encoding = %q, want %q", accepted, want)
	}
}

// Every encoding we advertise must survive the round trip, on every dialect.
func TestCompressedResponsesAreDecoded(t *testing.T) {
	cases := []string{"gzip", "deflate-zlib", "deflate-raw", "br", "zstd"}
	for _, enc := range cases {
		t.Run(enc, func(t *testing.T) {
			up, _ := encodingUpstream(t, enc,
				`{"choices":[{"message":{"content":"decoded"},"finish_reason":"stop"}],
				  "usage":{"prompt_tokens":11,"completion_tokens":3}}`)
			router := encodingRouter(t, up)

			// The Anthropic dialect reads the body with the strictest parser, so a
			// body that was not decoded properly fails here rather than passing as
			// an empty answer.
			resp, raw := postAnthropic(t, router.URL, "/v1/messages",
				`{"model":"mock-model","max_tokens":16,"messages":[{"role":"user","content":"hi"}]}`)
			if resp.StatusCode != 200 {
				t.Fatalf("status = %d, want 200: %s", resp.StatusCode, raw)
			}
			var out struct {
				Content []struct{ Text string } `json:"content"`
				Usage   struct {
					Input int `json:"input_tokens"`
				} `json:"usage"`
			}
			if err := json.Unmarshal(raw, &out); err != nil {
				t.Fatalf("decode: %v (%s)", err, raw)
			}
			if len(out.Content) == 0 || out.Content[0].Text != "decoded" {
				t.Errorf("content = %+v, want the decoded body", out.Content)
			}
			// Token accounting comes from the same body: a body that decoded only
			// partially would lose the usage block and silently stop billing.
			if out.Usage.Input != 11 {
				t.Errorf("input_tokens = %d, want 11 from the decoded usage block", out.Usage.Input)
			}
		})
	}
}

// Streaming must stay streaming through a decoder: buffering the whole body first
// would turn a live answer into a long silence.
func TestCompressedStreamingIsProgressive(t *testing.T) {
	sse := strings.Join([]string{
		`data: {"choices":[{"delta":{"content":"Hel"}}]}`,
		`data: {"choices":[{"delta":{"content":"lo"}}]}`,
		`data: {"choices":[{"delta":{},"finish_reason":"stop"}]}`,
		`data: {"choices":[],"usage":{"prompt_tokens":5,"completion_tokens":2}}`,
		`data: [DONE]`, "",
	}, "\n\n")

	up, _ := encodingUpstream(t, "br", sse)
	router := encodingRouter(t, up)

	_, raw := postAnthropic(t, router.URL, "/v1/messages",
		`{"model":"mock-model","max_tokens":16,"stream":true,"messages":[{"role":"user","content":"hi"}]}`)

	events := parseSSE(t, raw)
	var text string
	var sawStop bool
	for _, ev := range events {
		if ev.Name == "content_block_delta" {
			if d, ok := ev.Data["delta"].(map[string]any); ok && d["type"] == "text_delta" {
				text += d["text"].(string)
			}
		}
		if ev.Name == "message_stop" {
			sawStop = true
		}
	}
	if text != "Hello" {
		t.Errorf("streamed text = %q, want Hello through a brotli stream", text)
	}
	if !sawStop {
		t.Errorf("no message_stop: events = %v", events)
	}
}

// An encoding the router cannot read must fail loudly, not answer with an empty
// body that reads as a model that said nothing.
func TestUnknownEncodingFailsLoudly(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Encoding", "compress")
		w.WriteHeader(200)
		fmt.Fprint(w, "not really compressed")
	}))
	t.Cleanup(srv.Close)
	router := encodingRouter(t, srv)

	resp, raw := postAnthropic(t, router.URL, "/v1/messages",
		`{"model":"mock-model","max_tokens":16,"messages":[{"role":"user","content":"hi"}]}`)
	if resp.StatusCode == 200 {
		t.Fatalf("status = 200 for an undecodable body: %s", raw)
	}
	if !strings.Contains(string(raw), "compress") {
		t.Errorf("error should name the encoding it could not handle: %s", raw)
	}
}

// After decoding, the response must not still claim to be compressed: a reader that
// trusts the header — including this router's own accounting paths — would try to
// decode an already-decoded body.
func TestTransportClearsEncodingHeaders(t *testing.T) {
	body := compressWith(t, "gzip", []byte(`{"ok":true}`))
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Encoding", "gzip")
		w.Header().Set("Content-Length", fmt.Sprint(len(body)))
		w.Write(body)
	}))
	t.Cleanup(srv.Close)

	client := &http.Client{Transport: &decompressTransport{base: http.DefaultTransport}}
	resp, err := client.Get(srv.URL)
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	got, _ := io.ReadAll(resp.Body)

	if string(got) != `{"ok":true}` {
		t.Errorf("body = %q, want the decoded JSON", got)
	}
	if enc := resp.Header.Get("Content-Encoding"); enc != "" {
		t.Errorf("Content-Encoding still says %q", enc)
	}
	if resp.ContentLength != -1 {
		t.Errorf("ContentLength = %d, want -1 after decoding", resp.ContentLength)
	}
}

// A body that arrives uncompressed must pass through untouched.
func TestIdentityEncodingPassesThrough(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		fmt.Fprint(w, `{"choices":[{"message":{"content":"plain"},"finish_reason":"stop"}],
			"usage":{"prompt_tokens":2,"completion_tokens":1}}`)
	}))
	t.Cleanup(srv.Close)
	router := encodingRouter(t, srv)

	resp, raw := postAnthropic(t, router.URL, "/v1/messages",
		`{"model":"mock-model","max_tokens":16,"messages":[{"role":"user","content":"hi"}]}`)
	if resp.StatusCode != 200 || !strings.Contains(string(raw), "plain") {
		t.Fatalf("status = %d: %s", resp.StatusCode, raw)
	}
}

// A caller that sets its own Accept-Encoding keeps control.
func TestCallerAcceptEncodingIsRespected(t *testing.T) {
	up, seen := encodingUpstream(t, "identity",
		`{"choices":[{"message":{"content":"hi"}}],"usage":{"prompt_tokens":1,"completion_tokens":1}}`)
	client := &http.Client{Transport: &decompressTransport{base: http.DefaultTransport}}

	req, _ := http.NewRequest("GET", up.URL+"/x", nil)
	req.Header.Set("Accept-Encoding", "identity")
	resp, err := client.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	io.Copy(io.Discard, resp.Body)
	resp.Body.Close()

	accepted, _ := seen()
	if accepted != "identity" {
		t.Errorf("Accept-Encoding = %q, want the caller's own value", accepted)
	}
}
