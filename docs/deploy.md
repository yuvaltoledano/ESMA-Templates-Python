# Deploying ESMA-MILAN on a server

This guide puts the ESMA-MILAN web app on one Linux server, reachable
from any browser over HTTPS at a hostname of your choosing, protected by
a username and password. Follow it top to bottom the first time; the
"Day-to-day" and "If something goes wrong" sections are for later.

How it fits together, in one paragraph: the server runs two small
isolated programs ("containers") managed by Docker. One, called `app`,
is the Python service you run locally with `uv run esma-milan-server`.
The other, called `web`, is Caddy, a web server that handles the
encryption certificate, asks for the password, serves the page, and
passes the page's API requests on to `app`. Only `web` is reachable
from the internet. Everything that is specific to your server lives in
one file on the server, `.env`, which is never committed to git.

## 1. What you need before starting

- **A domain name and a hostname.** For example, if you own
  `example.com`, decide on `milan.example.com`. In your domain
  registrar's DNS settings, create an `A` record for that hostname
  pointing at the server's public IPv4 address (next item). Do this
  first: DNS changes can take up to an hour to spread, and the
  certificate step below needs the record to be live.
- **A server.** A Hetzner Cloud server (or any provider) running Ubuntu
  24.04, with your SSH public key added when you create it, and a
  firewall that allows inbound TCP on ports 22 (SSH), 80 (HTTP) and 443
  (HTTPS) and nothing else. 2 vCPUs and 4 GB of RAM are enough; 40 GB of
  disk is plenty. Note its public IPv4 address.
- **Your Mac's Terminal** (Applications, Utilities, Terminal). Every
  command below is typed there, either on the Mac or, after `ssh`, on
  the server. Lines starting with `$` are commands; do not type the `$`.

## 2. One-time server preparation

Log in to the server (replace `203.0.113.10` with your server's address;
answer `yes` to the fingerprint question the first time):

```
$ ssh root@203.0.113.10
```

Bring Ubuntu up to date:

```
# apt update && apt upgrade -y
```

Install Docker Engine and the Compose plugin from Docker's own package
repository. These are the steps from Docker's Ubuntu installation page
(https://docs.docker.com/engine/install/ubuntu/, "Install using the apt
repository"); if that page shows different commands, prefer the page.

```
# apt-get install -y ca-certificates curl
# install -m 0755 -d /etc/apt/keyrings
# curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
# chmod a+r /etc/apt/keyrings/docker.asc
# echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu $(. /etc/os-release && echo "${UBUNTU_CODENAME:-$VERSION_CODENAME}") stable" > /etc/apt/sources.list.d/docker.list
# apt-get update
# apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
```

Check that it worked. You should see version numbers, not an error:

```
# docker --version
Docker version 28.x.x, build ...
# docker compose version
Docker Compose version v2.x.x
```

Install `git` if it is not already there:

```
# apt-get install -y git
```

## 3. Get the code

Clone the repository into `/opt/esma-milan`. The `--recurse-submodules`
part also fetches the `r_reference` folder, which the build needs for
the ESMA taxonomy file:

```
# git clone --recurse-submodules https://github.com/yuvaltoledano/ESMA-Templates-Python.git /opt/esma-milan
# cd /opt/esma-milan
```

Trying a branch before it is merged? Clone that branch instead, and
switch back to `main` when it has been merged:

```
# git clone -b <branch-name> --recurse-submodules https://github.com/yuvaltoledano/ESMA-Templates-Python.git /opt/esma-milan
# cd /opt/esma-milan
...
# git checkout main && git pull --recurse-submodules && docker compose up -d --build
```

## 4. Configure

Create your `.env` from the example:

```
# cp .env.example .env
```

Generate the password hash. The command builds the `web` image the first
time (one to two minutes), then asks you to type a password twice and
prints one long line starting with `$2a$`:

```
# docker compose run --rm web caddy hash-password
Enter password:
Confirm password:
$2a$14$Q9...long string of letters, digits, . and /...
```

Choose a long random password (a password manager's generator is ideal)
and keep it in the password manager; the server stores only the hash.

Now edit `.env` with `nano` (arrow keys to move, Ctrl+O then Enter to
save, Ctrl+X to exit):

```
# nano .env
```

Fill in each line:

| Line | What to put |
|---|---|
| `SITE_HOST=` | your hostname, e.g. `milan.example.com` (no `https://`) |
| `BASIC_AUTH_USER=` | the username you will type in the browser |
| `BASIC_AUTH_HASH=` | the `$2a$...` line from above, **inside single quotes**: `BASIC_AUTH_HASH='$2a$14$Q9...'`. The quotes matter: without them Docker misreads the `$` signs and the password is never accepted. |
| `ESMA_MILAN_WORKERS=` | `2` for a 2-vCPU server; one worker handles one pool at a time |
| `ESMA_MILAN_RATE_LIMIT_PROCESS=` | leave `10/minute` unless you hit "Rate limit exceeded" |
| `ESMA_MILAN_RATE_LIMIT_HEALTH=` | leave `60/minute` |

Check that Docker reads the file the way you expect. The command prints
the full configuration; find `BASIC_AUTH_HASH` in it. It should show
your hash with every `$` doubled (`$$2a$$14$$...`), which is how Docker
displays a literal `$`. If it shows a truncated hash or warnings such as
`The "Q9" variable is not set`, the quotes are missing:

```
# docker compose config | grep BASIC_AUTH_HASH
```

## 5. First start

Build both images and start them. The first build downloads the base
images and compiles the frontend; expect five to ten minutes on a small
server, with a lot of scrolling output. It ends with lines like
`Container esma-milan-app-1  Healthy` and `Container esma-milan-web-1  Started`:

```
# docker compose up -d --build
```

Check that both containers are running and `app` is healthy:

```
# docker compose ps
NAME               IMAGE              STATUS                    PORTS
esma-milan-app-1   esma-milan-app     Up 2 minutes (healthy)    8000/tcp
esma-milan-web-1   esma-milan-web     Up 1 minute               0.0.0.0:80->80/tcp, 0.0.0.0:443->443/tcp
```

Watch Caddy obtain the certificate (Ctrl+C to stop watching; the
containers keep running). Within a minute you should see lines
containing `certificate obtained successfully` for your hostname:

```
# docker compose logs -f web
```

Now open `https://milan.example.com` (your hostname) in a browser on
your Mac. The browser asks for the username and password from `.env`.
The page appears, and the footer's health indicator shows "healthy"
within a few seconds.

Smoke test with the synthetic fixture that ships in the repository.
Download the two CSVs to your Mac from GitHub (open each link, then
"Download raw file"):

- `tests/fixtures/synthetic/loans.csv`
- `tests/fixtures/synthetic/collaterals.csv`

In the page, choose them as the loans and collaterals files, enter any
deal name (letters, digits, spaces, `.`, `-`, `_`), leave the taxonomy
empty (the server has the default one) and click **Process Pool &
Download**. A workbook named `2024-06-30 <deal> Flattened loans and
collaterals.xlsx` downloads within a few seconds. Try **Analyze Pool**
too.

## 6. Updating to a new version

When a new version is merged on GitHub:

```
# cd /opt/esma-milan
# git pull --recurse-submodules
# docker compose up -d --build
```

The running containers keep serving while the new images build; they are
replaced only once the build has finished, so the gap is a few seconds.
If nothing changed in the code, `docker compose up -d --build` finishes
quickly and leaves the containers running.

## 7. Day-to-day

All commands assume `cd /opt/esma-milan` first.

- **See what the app is doing** (Ctrl+C to stop watching):
  `docker compose logs -f app`. Each processed pool logs its deal name,
  file names, row counts and the output file name. Caddy's access log:
  `docker compose logs -f web`.
- **Restart everything:** `docker compose restart`.
- **Stop everything:** `docker compose down`. Certificates and settings
  are kept; `docker compose up -d` brings it back.
- **Change the password:** run `docker compose run --rm web caddy hash-password`
  again, put the new hash in `.env` (single quotes), then
  `docker compose up -d`. Caddy restarts in a second; browsers will ask
  for the new password on their next request.
- **Change the username or hostname:** edit `.env`, then
  `docker compose up -d`. A new hostname needs its DNS record first.
- **Change the rate limits or worker count:** edit `.env`, then
  `docker compose up -d`.
- **Check disk space:** `df -h /` for the system, `docker system df` for
  Docker's images. Old images from previous builds accumulate; reclaim
  them with `docker image prune -f` (keeps the images in use).
- **Where the certificates live:** in a Docker volume named
  `esma-milan_caddy_data`, renewed automatically by Caddy about a month
  before expiry. You never handle them. `docker compose down -v` would
  delete that volume; do not add `-v` unless you want a fresh start.

## 8. If something goes wrong

- **The browser says the site cannot be reached, or the certificate is
  invalid, and `docker compose logs web` shows errors mentioning
  `challenge` or `acme`.** Caddy could not prove to Let's Encrypt that it
  owns the hostname. Check that `SITE_HOST` is exactly the hostname in
  DNS, that the DNS `A` record points at this server (on your Mac:
  `dig +short milan.example.com` must print the server's IP), and that
  the firewall allows ports 80 and 443. Fix the cause, then wait: Caddy
  retries on its own. Do not restart repeatedly; Let's Encrypt limits
  failed attempts per hostname per hour.
- **The page loads but shows "Cannot reach API at localhost:8000".**
  The page shows this text for any error that did not come from the app
  itself, so it also appears when Caddy returns an error. Check
  `docker compose ps`: if `app` is not `(healthy)`, read
  `docker compose logs app` for the reason. If `app` is healthy, the
  request was probably rejected by Caddy before reaching the app: an
  upload set larger than 160 MB in total, or a session whose password
  expired in the browser (reload the page and log in again).
- **A `502` error, or `web` keeps restarting.** `app` is not healthy
  (see above), or the two containers started in the wrong order;
  `docker compose restart web` once `app` shows healthy.
- **"Uploaded file 'loans' exceeds the per-file size limit of 50 MB".**
  The app's own limit; it applies per file. Pools near that size are
  also near the 30,000-loan limit below.
- **"Pool size exceeds the synchronous endpoint limit".** The app refuses
  loans files with more than about 30,000 data rows, because a single
  request would take too long. Background processing of larger pools is
  planned; until then, split the pool.
- **A big pool runs for minutes and then the page shows an error.**
  Processing is synchronous: the browser waits for the whole run. The
  server side allows up to 20 minutes per request, which covers the
  30,000-loan limit, but a laptop that goes to sleep mid-request loses
  the download. Keep the Mac awake, or process smaller pools.
- **"Rate limit exceeded".** More than 10 pools per minute from one
  address (times the number of workers). Wait a minute, or raise
  `ESMA_MILAN_RATE_LIMIT_PROCESS` in `.env` and `docker compose up -d`.
- **The password is never accepted, and `docker compose logs web` shows
  nothing useful.** Almost always the missing single quotes around
  `BASIC_AUTH_HASH` in `.env`; see section 4 and re-run
  `docker compose config | grep BASIC_AUTH_HASH`.
- **`docker compose up` says `.env` is missing.** You are not in
  `/opt/esma-milan`, or section 4 was skipped.

## 9. Security notes

- The password protects every page and API call. The browser sends it
  with each request, but only inside the encrypted HTTPS connection, so
  it cannot be read on the way. Plain `http://` requests are redirected
  to HTTPS before any password is sent.
- Use a long random password and keep it in a password manager. The
  server stores a hash from which the password cannot be recovered.
- The `app` container has no port open to the internet; the only way in
  is through Caddy, after the password check. SSH (port 22) is the only
  other open port, and it accepts your key, not a password.
- Uploaded CSV files are written to a temporary folder for the duration
  of one request and deleted when it ends. The output workbook is built
  in memory and sent to the browser; it is never written to the server's
  disk. Nothing from a pool stays on the server after the response.
- The logs record deal names, file names, row counts, timings and the
  requesting address. They do not contain loan data.
- Keep the server updated: log in monthly and run
  `apt update && apt upgrade -y`, then `reboot` if a kernel update asks
  for it (the containers come back on their own). Consider
  `apt install unattended-upgrades` so security patches install
  themselves.
- Keep the Docker images current too: `docker compose up -d --build`
  after a `git pull` rebuilds on the latest base images.

## 10. Resizing the server later

If pools take too long or two analysts often process at once, resize the
server at the provider (Hetzner: power off, Rescale, pick a bigger type,
power on). Nothing in the app changes. After the resize, set
`ESMA_MILAN_WORKERS` in `.env` to the new number of vCPUs and run
`docker compose up -d`. Each worker handles one pool at a time; a
10,000-loan pool measured about 30 seconds and 200 MB of memory per
worker, so budget roughly 1 GB of RAM per worker to stay comfortable at
the 30,000-loan limit.
