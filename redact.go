package main

import (
	"strings"
)

// shortErr memendekkan ralat provider sebelum ia masuk ke jurnal.
//
// Badan respons pihak atas kadangkala memetik semula sebahagian parameter
// permintaan. Log hanya perlukan sebab kegagalan, bukan kandungan itu, jadi
// kita simpan status dan permulaan mesej sahaja.
func shortErr(err error) string {
	if err == nil {
		return ""
	}

	s := strings.ReplaceAll(err.Error(), "\n", " ")
	s = strings.Join(strings.Fields(s), " ")

	const max = 120
	if len(s) > max {
		cut := max
		for cut > 0 && !utf8Start(s[cut]) {
			cut--
		}
		s = s[:cut] + "…"
	}
	return s
}

// utf8Start melaporkan sama ada bait pada kedudukan i memulakan aksara UTF-8.
func utf8Start(b byte) bool {
	return b&0xC0 != 0x80
}
