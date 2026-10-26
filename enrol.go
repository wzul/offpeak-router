package main

// Key enrolment.
//
// Adding a credential to the router means writing one line into /opt/offpeak-router/.env,
// and the person who holds the key should not have to paste it into a chat, a shell
// history or a ticket to get it there. This endpoint exists for exactly one paste:
//
//	GET  /enrol/<token>   a form with one field
//	POST /enrol/<token>   writes OLLAMA_LENJAN_API_KEY and exits so systemd reloads it
//
// It is off unless a token file exists, the token is single-use, and it only ever writes
// one known key name. The response never echoes the key back.
//
// The service cannot call systemctl (NoNewPrivileges, ProtectSystem=strict), so it writes
// the file and exits; Restart=always brings it back with the new EnvironmentFile. That is
// the only restart path available from inside this unit, and it is enough.

import (
	"crypto/rand"
	"crypto/subtle"
	"encoding/hex"
	"fmt"
	"html"
	"log"
	"net/http"
	"os"
	"strings"
	"sync"
	"time"
)

const (
	enrolKeyName = "OLLAMA_LENJAN_API_KEY"
	enrolMaxBody = 8 << 10
)

// Paths are variables so a test can point them at a temporary directory: the point of the
// test is that the write is one-shot and safe, and doing that against the real
// /opt/offpeak-router/.env would be an accident waiting to happen.
var (
	enrolTokenPathVar = "/opt/offpeak-router/enrol.token"
	enrolEnvPathVar   = "/opt/offpeak-router/.env"
	// exitProcess ends the process so systemd reloads the EnvironmentFile. A variable so
	// a test can observe it instead of dying.
	exitProcess = os.Exit
)

var enrolMu sync.Mutex

// enrolToken reads the live token, if enrolment is open.
func enrolToken() string {
	buf, err := os.ReadFile(enrolTokenPathVar)
	if err != nil {
		return ""
	}
	return strings.TrimSpace(string(buf))
}

// mintEnrolToken opens enrolment and returns the token to put in the URL.
func mintEnrolToken() (string, error) {
	raw := make([]byte, 24)
	if _, err := rand.Read(raw); err != nil {
		return "", err
	}
	token := hex.EncodeToString(raw)
	if err := os.WriteFile(enrolTokenPathVar, []byte(token+"\n"), 0o600); err != nil {
		return "", err
	}
	return token, nil
}

func handleEnrol(w http.ResponseWriter, r *http.Request) {
	live := enrolToken()
	if live == "" {
		// No enrolment open: indistinguishable from a route that does not exist, so a
		// probe cannot tell whether this feature is installed.
		http.NotFound(w, r)
		return
	}
	asked := strings.TrimPrefix(r.URL.Path, "/enrol/")
	if subtle.ConstantTimeCompare([]byte(asked), []byte(live)) != 1 {
		http.NotFound(w, r)
		return
	}

	switch r.Method {
	case http.MethodGet:
		w.Header().Set("Content-Type", "text/html; charset=utf-8")
		w.Header().Set("Cache-Control", "no-store")
		fmt.Fprint(w, enrolForm("", ""))
		return
	case http.MethodPost:
	default:
		w.Header().Set("Allow", "GET, POST")
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}

	r.Body = http.MaxBytesReader(w, r.Body, enrolMaxBody)
	if err := r.ParseForm(); err != nil {
		writeEnrolPage(w, http.StatusBadRequest, "Too large or malformed. Paste only the key.")
		return
	}
	key := strings.TrimSpace(r.PostFormValue("key"))
	if key == "" {
		writeEnrolPage(w, http.StatusBadRequest, "Empty key: nothing was written.")
		return
	}
	if len(key) > 512 || strings.ContainsAny(key, "\r\n\t ") {
		// A key with whitespace or newlines is a paste accident, and writing it would
		// produce an EnvironmentFile line systemd misreads.
		writeEnrolPage(w, http.StatusBadRequest,
			"That does not look like a single key. Nothing was written.")
		return
	}

	enrolMu.Lock()
	defer enrolMu.Unlock()

	// Single use: the token is consumed before the write, so a replay cannot land even
	// if the response is somehow repeated.
	if err := os.Remove(enrolTokenPathVar); err != nil {
		writeEnrolPage(w, http.StatusConflict, "Enrolment already used.")
		return
	}

	if err := writeEnvValue(enrolEnvPathVar, enrolKeyName, key); err != nil {
		log.Printf("enrol: write failed: %v", err)
		writeEnrolPage(w, http.StatusInternalServerError, "Could not write the file on the server.")
		return
	}
	log.Printf("enrol: %s written; exiting so systemd reloads the environment", enrolKeyName)

	writeEnrolPage(w, http.StatusOK,
		"Saved. The router restarts in a few seconds and "+enrolKeyName+" becomes active.")
	// Flush before exiting, or the browser may see a reset instead of the confirmation.
	if f, ok := w.(http.Flusher); ok {
		f.Flush()
	}
	// The exit is captured now, not read after the sleep: a test that restores this hook
	// when it finishes would otherwise let the delayed call hit os.Exit anyway.
	go func(exit func(int)) {
		time.Sleep(300 * time.Millisecond)
		exit(0)
	}(exitProcess)
}

// writeEnvValue sets one key in a systemd EnvironmentFile, preserving every other line and
// the file mode. Writing through a temporary file keeps a partial write from taking the
// service down with a truncated environment.
func writeEnvValue(path, name, value string) error {
	existing, err := os.ReadFile(path)
	if err != nil && !os.IsNotExist(err) {
		return err
	}
	lines := strings.Split(strings.TrimRight(string(existing), "\n"), "\n")
	if len(lines) == 1 && lines[0] == "" {
		lines = nil
	}

	replaced := false
	for i, line := range lines {
		if strings.HasPrefix(line, name+"=") {
			lines[i] = name + "=" + value
			replaced = true
		}
	}
	if !replaced {
		lines = append(lines, name+"="+value)
	}

	tmp := path + ".tmp"
	body := strings.Join(lines, "\n") + "\n"
	if err := os.WriteFile(tmp, []byte(body), 0o600); err != nil {
		return err
	}
	if err := os.Chmod(tmp, 0o600); err != nil {
		return err
	}
	return os.Rename(tmp, path)
}

func writeEnrolPage(w http.ResponseWriter, code int, message string) {
	w.Header().Set("Content-Type", "text/html; charset=utf-8")
	w.Header().Set("Cache-Control", "no-store")
	w.WriteHeader(code)
	fmt.Fprint(w, enrolForm(message, ""))
}

func enrolForm(message, token string) string {
	note := ""
	if message != "" {
		note = `<p class="note">` + html.EscapeString(message) + `</p>`
	}
	return `<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Router key enrolment</title>
<style>
 body{font:16px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;
      max-width:34rem;margin:8vh auto;padding:0 1.25rem;color:#141414;background:#fafafa}
 h1{font-size:1.15rem;margin:0 0 .25rem}
 p{color:#555;margin:.4rem 0 1.1rem}
 .note{color:#8a2b00;font-weight:600}
 label{display:block;font-size:.85rem;letter-spacing:.02em;text-transform:uppercase;color:#666}
 input{width:100%;padding:.75rem;margin:.4rem 0 1rem;font:inherit;border:1px solid #ccc;
       border-radius:.5rem;background:#fff}
 button{padding:.7rem 1.2rem;font:inherit;font-weight:600;color:#fff;background:#141414;
        border:0;border-radius:.5rem;cursor:pointer}
 small{display:block;margin-top:1.4rem;color:#777}
</style></head><body>
<h1>Ollama key enrolment</h1>
<p>Paste the API key for the spare account. It is written to the router's environment file
(mode 600) and the router restarts itself; this link works once.</p>` + note + `
<form method="post">
 <label for="key">API key</label>
 <input id="key" name="key" type="password" autocomplete="off" spellcheck="false"
        autocapitalize="off" placeholder="paste the key" required>
 <button type="submit">Save and restart</button>
</form>
<small>The key is never sent back to the browser, logged, or shown again.</small>
</body></html>`
}
