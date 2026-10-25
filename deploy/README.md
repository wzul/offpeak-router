# Deploying behind a reverse proxy

The router binds to loopback by default and has no transport security of its
own. Put it behind a TLS-terminating proxy whenever a client is not on the
same host, and set `ROUTER_AUTH_TOKEN` regardless.

## systemd

`offpeak-router.service` in the repo root is a hardened unit: read-only
filesystem except the working directory, `MemoryMax=64M`, `GOGC=50`.
Install it, put the environment in an `EnvironmentFile` (mode `0600`), and
point `ExecStart` at the binary.

```bash
install -Dm755 offpeak-router /opt/offpeak-router/offpeak-router
install -Dm644 offpeak-router.service /etc/systemd/system/offpeak-router.service
install -Dm600 /dev/stdin /opt/offpeak-router/.env <<'ENV'
ROUTER_ADDR=127.0.0.1:8788
ROUTER_DB=/opt/offpeak-router/spend.json
ROUTER_AUTH_TOKEN=<generate-with-openssl-rand-hex-32>
OLLAMA_API_KEY=<your-key>
NOUS_API_KEY=<your-key>
ENV
systemctl daemon-reload && systemctl enable --now offpeak-router
```

If the proxy runs in a Docker container, bind the router to the bridge
gateway (`172.17.0.1:8788`) instead of loopback, and restrict the port at the
firewall to the container network only.

## Reverse proxy

See `nginx.conf.example`. Two lines matter more than the rest:

- **`proxy_buffering off`** — otherwise SSE is buffered until the completion
  finishes and streaming stops working.
- **an ACME location** — nginx-proxy with `HTTPS_METHOD=nohttps` does not
  inject one, so `/.well-known/acme-challenge/` must be answered by your own
  server block or certificate issuance returns 404.

## Gotchas that cost real time

- **Cloudflare in front of a provider.** At least one upstream sits behind
  Cloudflare and answers **403** to Go's default `Go-http-client/1.1` and to
  an empty User-Agent, while accepting any identifiable UA. The router always
  sends `offpeak-router/<version>`; do not strip it in the proxy.
- **"Always Use HTTPS" breaks HTTP-01 issuance.** If the zone forces HTTP to
  HTTPS at the edge, the ACME challenge is redirected before reaching the
  origin and Let's Encrypt reports `404` (after following the redirect to a
  vhost that has no certificate yet). Either publish the record **DNS-only**
  for the API hostname, or issue a Cloudflare Origin Certificate instead of
  using ACME.
- **Never reference a certificate that does not exist yet.** With
  nginx-proxy, a `CERT_NAME` pointing at a missing file makes it emit a broken
  server block, `nginx -t` fails, and the reload takes down every other vhost.
  Create HTTP-only first, issue, then add `CERT_NAME` and switch to redirect.
