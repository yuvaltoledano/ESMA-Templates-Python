# Deploying ESMA-MILAN on a server

This guide puts the ESMA-MILAN web app on one Linux server, reachable
from any browser over HTTPS at a hostname of your choosing, protected by
a username and password. Follow it top to bottom the first time; the
"Day-to-day" and "If something goes wrong" sections are for later.

How it fits together, in one paragraph: the server runs three small
isolated programs ("containers") managed by Docker. `app` is the Python
service you run locally with `uv run esma-milan-server`; it answers the
page's requests. `worker` runs the pools the page submits, one at a
time, in the background, so a very large pool can take a quarter of an
hour without anyone waiting on an open connection (section 11). `web`
is Caddy, a web server that handles the encryption certificate, asks
for the password, serves the page, and passes the page's API requests
on to `app`. Only `web` is reachable from the internet. Everything that
is specific to your server lives in one file on the server, `.env`,
which is never committed to git.

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

If `apt-get update` reports that the Docker repository has no Release
file (this happens on a very new Ubuntu release that Docker does not
package yet), replace the `$(. /etc/os-release && echo ...)` expression
in the `echo "deb ..."` line with `noble` and run that line and the two
`apt-get` commands again.

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

Add a 2 GB swap file. A pool of a few hundred thousand loans needs most
of a 4 GB server's memory (section 11); without swap, running out means
the kernel kills the worker and the job fails. With swap, the job slows
down instead of dying. This is a safety net, not a substitute for
enough memory:

```
# fallocate -l 2G /swapfile
# chmod 600 /swapfile
# mkswap /swapfile
# swapon /swapfile
# echo '/swapfile none swap sw 0 0' >> /etc/fstab
# swapon --show
NAME      TYPE SIZE USED PRIO
/swapfile file   2G   0B   -2
```

The `fstab` line makes the swap file come back after a reboot.

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
# docker compose run --rm --no-deps web caddy hash-password
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
NAME                  IMAGE              STATUS                    PORTS
esma-milan-app-1      esma-milan-app     Up 2 minutes (healthy)    8000/tcp
esma-milan-web-1      esma-milan-web     Up 1 minute               0.0.0.0:80->80/tcp, 0.0.0.0:443->443/tcp
esma-milan-worker-1   esma-milan-app     Up 2 minutes
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
Download**. The page shows the job's progress ("Stage 7 of 11 ...") and,
within a few seconds, a **Download workbook** button for
`2024-06-30 <deal> Flattened loans and collaterals.xlsx`; click it, then
**Delete from server**. Try **Analyze Pool** too.

## 6. Updating to a new version

When a new version is merged on GitHub, first make sure no pool is
being processed: `docker compose up -d --build` restarts the worker, and
a job that is running at that moment is lost (it is marked "interrupted"
and has to be submitted again). This prints the running job's marker if
there is one, or "no job running":

```
# cd /opt/esma-milan
# docker compose exec worker sh -c 'ls /jobs/*/running.* 2>/dev/null || echo "no job running"'
no job running
```

Then:

```
# git pull --recurse-submodules
# docker compose up -d --build
```

The running containers keep serving while the new images build; they are
replaced only once the build has finished, so the gap is a few seconds.
If nothing changed in the code, `docker compose up -d --build` finishes
quickly and leaves the containers running. The first update to a version
with background jobs adds the `worker` container: `docker compose ps`
shows three containers afterwards instead of two, and a `jobs` volume
appears in `docker volume ls`.

## 7. Day-to-day

All commands assume `cd /opt/esma-milan` first.

- **See what the app is doing** (Ctrl+C to stop watching):
  `docker compose logs -f app` for the API (uploads, job creation,
  downloads) and `docker compose logs -f worker` for the pools being
  processed: each job logs `job_start`, its stages, and `job_done` with
  the duration or `job_failed` with the reason. Caddy's access log:
  `docker compose logs -f web`.
- **Restart everything:** `docker compose restart`.
- **Stop everything:** `docker compose down`. Certificates and settings
  are kept; `docker compose up -d` brings it back.
- **Change the password:** run `docker compose run --rm --no-deps web caddy hash-password`
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
- **"Pool size exceeds the synchronous endpoint limit".** Only
  `POST /api/process`, the old one-request path, has that limit. The page
  uses background jobs, which have none; if you see this, the request was
  made with a script or an old copy of the page. Reload the page.
- **The job fails with code `interrupted`.** The worker was restarted
  while the pool was running: either `docker compose up -d --build` was
  run during a job (see section 6), or the server ran out of memory and
  the kernel killed the worker. `docker compose logs worker` shows which;
  `dmesg | grep -i "out of memory"` confirms a memory kill. Add the swap
  file from section 2 if it is missing, resize the server for pools of
  that size (section 11), and submit the pool again.
- **The job fails with code `output_too_large`.** A sheet of the workbook
  would exceed Excel's 1,048,576-row limit (the message names the sheet),
  so nothing was written. Split the pool.
- **The page lost the job (closed tab, new browser).** The job keeps
  running. The page remembers the job only in the tab that started it;
  from another tab there is no list of jobs. Wait for the original tab,
  or submit the pool again and delete the duplicate result afterwards.
- **A job sits at "Queued" and never starts.** The worker is not running:
  `docker compose ps` should show `esma-milan-worker-1` as `Up`;
  `docker compose logs worker` shows why not; `docker compose up -d`
  starts it. Queued jobs are picked up as soon as it runs.
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
- Uploaded CSV files are written to the job's folder on the server for
  the duration of the run and deleted the moment the job finishes,
  whether it succeeded or failed. The output workbook (or the analysis
  result) stays on the server's disk, in the `jobs` volume, until you
  click **Delete from server** or until the retention period
  (`ESMA_MILAN_JOB_RETENTION_HOURS` in `.env`, 12 hours by default)
  passes, when the worker deletes it. Delete results you have downloaded;
  do not rely on retention for sensitive pools.
- The logs record deal names, job ids, file sizes, row counts, timings,
  stage names and the requesting address. They do not contain loan data.
- Keep the server updated: log in monthly and run
  `apt update && apt upgrade -y`, then `reboot` if a kernel update asks
  for it (the containers come back on their own). Consider
  `apt install unattended-upgrades` so security patches install
  themselves.
- Keep the Docker images current too: `docker compose up -d --build`
  after a `git pull` rebuilds on the latest base images.

## 10. Resizing the server later

If pools take too long, resize the server at the provider (Hetzner:
power off, Rescale, pick a bigger type, power on). Nothing in the app
changes. After the resize, set `ESMA_MILAN_WORKERS` in `.env` to the new
number of vCPUs and run `docker compose up -d`.

Memory is what decides the server size. Background jobs run one at a
time and the memory a job needs grows with the pool: measured about
200 MB at 10,000 loans, 470 MB at 50,000 and 780 MB at 100,000, so a pool
of around 300,000 loans needs about 3 GB. On the 4 GB server that fits,
with the swap file from section 2 as the safety net, but with little to
spare; if pools of that size are routine, use the 8 GB tier.

## 11. Large pools: background jobs

Every pool the page submits is a background job. The `app` container
stores the uploaded files in a job folder and answers immediately; the
`worker` container picks the job up, runs the pipeline, and writes the
workbook (or the analysis result) next to it; the page polls every two
seconds and shows the stage ("Stage 7 of 11: flatten and derived fields")
until the job is done, then offers **Download workbook** and **Delete
from server**. Closing the tab does not stop the job; reopening the same
tab resumes watching it. Jobs run one at a time, in the order they were
submitted; a second pool waits in the queue ("Queued, waiting for the
worker to start") until the first finishes.

**Time and memory.** A 10,000-loan pool takes about half a minute, a
100,000-loan pool about five minutes, and a 300,000-loan pool about a
quarter of an hour, almost all of it writing the workbook. Memory grows
with the pool too: about 3 GB at 300,000 loans (section 10). Keep the
swap file from section 2; if the kernel kills the worker for memory, the
job ends as `interrupted` and the pool must be submitted again.

**Where things live.** The `jobs` volume (`esma-milan_jobs`), mounted at
`/jobs` in both `app` and `worker`, holds one folder per job: the
uploaded files while the job runs (deleted when it finishes), the
status, and the result. Results stay until deleted from the page or
until `ESMA_MILAN_JOB_RETENTION_HOURS` (`.env`, default 12) after the
job finished; the worker sweeps expired jobs every minute.

**Disk space.** During a run the volume holds the uploaded files (up to
600 MB each by the `ESMA_MILAN_MAX_UPLOAD_MB` cap in `docker-compose.yml`;
a 300,000-loan pool is about 175 MB of loans and 65 MB of collaterals)
plus the workbook being written (about 430 MB at 300,000 loans); during
the upload itself the server briefly holds a second copy. Results that
are not deleted accumulate until retention expires. Keep at least 5 GB
free for jobs. To see what the volume holds:

```
# docker compose exec worker du -sh /jobs
```

To empty it completely (every queued, running and finished job is lost):

```
# docker compose down
# docker volume rm esma-milan_jobs
# docker compose up -d
```

**Watching the worker.** `docker compose logs -f worker` shows each job
start, stage by stage, and finish (`job_done` with the duration, or
`job_failed` with the error code). `docker compose restart worker`
restarts it; a job running at that moment ends as `interrupted`.

**Uploads.** The upload itself goes through your connection: a 175 MB
loans file takes a few minutes on a typical home upload speed, during
which the page shows "Uploading files...". Caddy allows requests up to
1900 MB (three files at the 600 MB cap).
