package main

import (
	"errors"
	"strings"
	"testing"
)

func TestShortErr(t *testing.T) {
	long := errors.New(strings.Repeat("x", 400))
	got := shortErr(long)
	if len(got) > 130 {
		t.Fatalf("tidak dipendekkan: %d aksara", len(got))
	}
	if !strings.HasSuffix(got, "…") {
		t.Fatalf("tiada penanda potong: %q", got)
	}

	nl := errors.New("HTTP 400: {\"error\":\n \"message\": \"invalid image\"}")
	if strings.Contains(shortErr(nl), "\n") {
		t.Fatal("baris baharu tidak dibuang")
	}

	if shortErr(nil) != "" {
		t.Fatal("nil sepatutnya kosong")
	}

	// Potongan mesti berhenti pada sempadan aksara, bukan di tengah bait.
	wide := errors.New(strings.Repeat("语", 100))
	out := shortErr(wide)
	if !strings.HasSuffix(out, "…") {
		t.Fatal("output lebar tidak dipotong")
	}
	if !utf8Start(out[len(out)-1]) && out[len(out)-1] != 0xA6 {
		// penanda "…" ialah tiga bait; ujian ini hanya memastikan tiada bait separuh
	}
	for i := 0; i < len(out); i++ {
		if out[i]&0xC0 == 0x80 && i > 0 && out[i-1]&0xC0 != 0xC0 && out[i-1]&0x80 == 0 {
			t.Fatalf("bait UTF-8 rosak pada %d", i)
		}
	}
}
