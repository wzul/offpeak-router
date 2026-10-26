package main

// Upstream Accept-Encoding, taken all the way.
//
// The router advertises the same compression set as the client it mimics, which
// means the upstream may answer in brotli or zstd. Go's transport only decompresses
// gzip, and it only does so when it set the header itself — so a transport wrapper
// does both jobs here: it sends the full set and decodes whatever comes back. Doing
// it in the RoundTripper means every route family (chat, decision, Ollama, Anthropic,
// token counting) gets it without touching its own body handling, and a new route
// cannot forget.

import (
	"bufio"
	"compress/flate"
	"compress/gzip"
	"compress/zlib"
	"fmt"
	"io"
	"net/http"
	"strings"

	"github.com/andybalholm/brotli"
	"github.com/klauspost/compress/zstd"
)

// upstreamAcceptEncoding is exactly what the measured client sends.
const upstreamAcceptEncoding = "gzip, deflate, br, zstd"

type decompressTransport struct {
	base http.RoundTripper
}

func (t *decompressTransport) RoundTrip(req *http.Request) (*http.Response, error) {
	// Only set it when the caller has not: a route that deliberately asks for no
	// compression (or a test) keeps control.
	if req.Header.Get("Accept-Encoding") == "" {
		req.Header.Set("Accept-Encoding", upstreamAcceptEncoding)
	}
	resp, err := t.base.RoundTrip(req)
	if err != nil || resp == nil {
		return resp, err
	}

	enc := strings.ToLower(strings.TrimSpace(resp.Header.Get("Content-Encoding")))
	if enc == "" || enc == "identity" {
		return resp, nil
	}
	body, err := decodeBody(resp.Body, enc)
	if err != nil {
		resp.Body.Close()
		return nil, fmt.Errorf("decode %s response: %w", enc, err)
	}
	resp.Body = body
	// The decoded length no longer matches, and a downstream reader that trusts
	// Content-Length would stop early.
	resp.Header.Del("Content-Encoding")
	resp.Header.Del("Content-Length")
	resp.ContentLength = -1
	return resp, nil
}

// decodeBody wraps the body with the right decoder. Encodings are applied in the
// order listed by the server, so they are unwound in reverse.
func decodeBody(body io.ReadCloser, encoding string) (io.ReadCloser, error) {
	parts := strings.Split(encoding, ",")
	out := body
	for i := len(parts) - 1; i >= 0; i-- {
		name := strings.ToLower(strings.TrimSpace(parts[i]))
		if name == "" || name == "identity" {
			continue
		}
		next, err := decoderFor(name, out)
		if err != nil {
			out.Close()
			return nil, err
		}
		out = next
	}
	return out, nil
}

func decoderFor(name string, body io.ReadCloser) (io.ReadCloser, error) {
	switch name {
	case "gzip", "x-gzip":
		r, err := gzip.NewReader(body)
		if err != nil {
			return nil, err
		}
		return &readCloserPair{Reader: r, closers: []io.Closer{r, body}}, nil
	case "deflate":
		// "deflate" in the wild is either zlib-wrapped or raw flate. Peek at the
		// first byte: a zlib header's low nibble is 8 and the two bytes form a
		// valid checksum, so 0x78 is the common case.
		br := bufio.NewReader(body)
		head, _ := br.Peek(1)
		if len(head) == 1 && head[0]&0x0f == 8 {
			r, err := zlib.NewReader(br)
			if err == nil {
				return &readCloserPair{Reader: r, closers: []io.Closer{r, body}}, nil
			}
		}
		r := flate.NewReader(br)
		return &readCloserPair{Reader: r, closers: []io.Closer{r, body}}, nil
	case "br":
		return &readCloserPair{Reader: brotli.NewReader(body), closers: []io.Closer{body}}, nil
	case "zstd":
		d, err := zstd.NewReader(body)
		if err != nil {
			return nil, err
		}
		return &readCloserPair{Reader: d, closers: []io.Closer{d.IOReadCloser(), body}}, nil
	}
	return nil, fmt.Errorf("unsupported content-encoding %q", name)
}

// readCloserPair joins a reader with the closers it depends on, so nothing leaks
// when a caller closes the decoded body.
type readCloserPair struct {
	io.Reader
	closers []io.Closer
}

func (r *readCloserPair) Close() error {
	var first error
	for _, c := range r.closers {
		if err := c.Close(); err != nil && first == nil {
			first = err
		}
	}
	return first
}
