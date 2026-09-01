# Go Live and Challenge Testing Guide

This guide takes the Banorte CV Agent from the current GitHub repository to a public HTTPS URL, then shows how to connect the **correct** endpoint in a challenge website without claiming protocol compatibility the project does not have.

## 1. Know what is ready—and what is not

The application is ready to deploy as one public FastAPI service. Its supported public interfaces are:

| Purpose | Method and path | Use it for |
|---|---|---|
| Browser demo | `GET /` | The built-in chat interface |
| Readiness | `GET /health` | Render health checks and basic availability tests |
| Agent chat | `POST /api/chat` | A custom REST client or a compatible challenge integration |

The agent chat endpoint is **not** an OpenAI Responses API, an A2A agent, or an MCP server. Do not enter its URL into a challenge field that requires one of those protocols. That would be a contract mismatch, even if the URL is publicly reachable.

## 2. Prepare the required secret

1. Sign in to Anthropic and create one API key for this project.
2. Store it only in Render’s environment variables as `ANTHROPIC_API_KEY`.
3. Never commit it to GitHub, put it in the challenge website, or expose it in frontend JavaScript.

The static UI and `/health` work without a key. Real chat answers require a valid key. This is the project's only runtime API key: GitHub and Render account access are not application API keys, and the Anthropic key must never be pasted into the challenge website.

## 3. Deploy with Render

Render is the project’s chosen initial host. It can deploy a linked GitHub repository as a Web Service, builds this repository’s existing `Dockerfile`, supplies `PORT`, exposes a public `onrender.com` URL, and supports HTTP health checks. See [Render Web Services](https://render.com/docs/web-services), [Docker on Render](https://render.com/docs/docker), and [Health Checks](https://render.com/docs/health-checks).

1. Open Render and choose **New** → **Web Service**.
2. Connect the GitHub account that owns `mdrt95/reto`, then select that repository and the `main` branch.
3. Choose the Docker runtime. Render will use the repository's existing `Dockerfile`; do not add a second start command.
4. In the service’s **Environment** settings, set:

   ```text
   ANTHROPIC_API_KEY=<your Anthropic key>
   ENVIRONMENT=production
   LOG_LEVEL=INFO
   ```

   The remaining settings have safe defaults. Only override them deliberately:

   ```text
   MODEL_NAME=claude-sonnet-4-6
   MODEL_TIMEOUT_SECONDS=30
   MAX_INPUT_CHARS=12000
   MAX_HISTORY_MESSAGES=12
   RATE_LIMIT_PER_MINUTE=30
   FORWARDED_ALLOW_IPS=*
   ```

   `FORWARDED_ALLOW_IPS` must stay `*` only because Render puts exactly one trusted proxy in front of this container; the app trusts that proxy's forwarded headers to see the real client IP for per-IP rate limiting.

5. In **Settings** → **Health Checks**, set the HTTP health-check path to `/health`.
6. Record the generated public domain as:

   ```text
   APP_URL=https://<your-service>.onrender.com
   ```

7. Wait for the deployment to become healthy. A Render HTTP health check succeeds only after the configured endpoint returns a 2xx or 3xx status.

## 4. Verify the public deployment before touching the challenge website

Run these commands from your terminal after replacing `APP_URL` with the generated HTTPS URL:

```bash
curl --fail --silent --show-error "$APP_URL/health"
```

Expected shape:

```json
{"version":"0.1.0","ready":true}
```

Then verify a real agent request:

```bash
curl --silent --show-error --request POST "$APP_URL/api/chat" \
  --header "Content-Type: application/json" \
  --data '{
    "message": "What security-related work has Marco done?",
    "history": [],
    "preferences": {"verbosity": "concise"}
  }'
```

Expected success shape:

```json
{
  "id": "req_<uuid>",
  "answer": "...",
  "conversation_id": "conv_<uuid>",
  "status": "completed"
}
```

If the response is a `503` with `generation_unavailable`, re-check the Render variable name and API-key validity. Do not expose the key while troubleshooting; use Render deployment logs, which must stay free of prompt content and secrets.

Then confirm the two boundary behaviors that must never regress:

```bash
curl --silent --show-error --request POST "$APP_URL/api/chat" \
  --header "Content-Type: application/json" \
  --data '{"message": "Tell me about Marco'\''s experience at Google.", "history": []}'
```

Expected: a not-found-style answer stating that Google experience is not in Marco's profile — never an answer built from unrelated employers or projects.

```bash
curl --silent --show-error --request POST "$APP_URL/api/chat" \
  --header "Content-Type: application/json" \
  --data '{"message": "How can I contact Marco?", "history": []}'
```

Expected: a safe redirect reply that does not include an email address or phone number.

Before calling the release complete, run the credentialed fixed evaluation from a safe environment that has the same key configured:

```bash
python -m eval.run_eval --execute
```

The GitHub Actions workflow intentionally runs only the offline checks; it never receives the Anthropic key.

## 5. Find the right field in the challenge website

Open the challenge’s testing or integration page and first identify **what contract it expects**. Use its published docs, field help text, request example, or browser developer tools → **Network** after sending a sample test. Record these four facts:

1. The HTTP method and path the website will call.
2. The JSON request body it will send.
3. The JSON response body it expects.
4. Whether its request originates server-to-server or from the browser.

Use this mapping:

| Challenge field or contract | What to enter / do | Compatible now? |
|---|---|---:|
| **Chat endpoint**, **Webhook URL**, or a custom REST `POST` endpoint | Enter the complete URL: `https://<your-domain>/api/chat` | Yes, if its body and response match the contract below |
| **Base URL** and docs explicitly say the client appends `/api/chat` | Enter only `https://<your-domain>` | Yes |
| **Demo URL**, **Agent website**, or presentation link | Enter `https://<your-domain>/` | Yes; this opens the built-in frontend |
| **Health-check URL** | Enter `https://<your-domain>/health` | Yes, but it cannot answer chat questions |
| **OpenAI base URL** or **`/v1/responses`** | Do not connect it yet | No—requires a Responses-style adapter |
| **A2A agent URL**, **Agent Card**, or **`/.well-known/agent-card.json`** | Do not connect it yet | No—requires an A2A adapter and agent card |
| **MCP server URL** | Do not connect it yet | No—requires an MCP transport/server adapter |

The exact field name and required protocol belong to the challenge website. If its UI or docs do not make them clear, capture the request example or a screenshot of the integration form before changing this code. A protocol adapter should be selected only after that evidence is available.

## 6. Validate a compatible custom REST connection

The current chat contract requires:

```http
POST /api/chat
Content-Type: application/json
```

```json
{
  "message": "A non-empty question",
  "history": [
    {"role": "user", "content": "Earlier question"},
    {"role": "assistant", "content": "Earlier answer"}
  ],
  "preferences": {
    "language": "en",
    "verbosity": "concise"
  }
}
```

Only `message` is required. `history` and `preferences` are optional. Do not add arbitrary fields: request validation rejects them. The successful response must contain `answer` and `status: "completed"` as shown above.

### Browser-origin request versus server-origin request

The built-in frontend is same-origin and needs no CORS configuration. If the challenge website performs browser-side JavaScript requests from a different origin, the browser will block this API until the exact challenge origin is allowlisted through CORS.

Do **not** use `Access-Control-Allow-Origin: *` by default. First obtain the challenge page’s exact origin, then add a narrowly scoped CORS configuration, a focused regression test, and redeploy. A server-to-server challenge integration does not need CORS.

## 7. Final go-live checklist

- [ ] GitHub Actions `Verify` workflow is green for the deployed commit.
- [ ] `ANTHROPIC_API_KEY` is set only in Render.
- [ ] Render health check uses `/health` and returns a successful status.
- [ ] The generated HTTPS root URL loads the chat frontend.
- [ ] The `curl` request to `POST /api/chat` returns a grounded answer.
- [ ] `python -m eval.run_eval --execute` has been run with the deployment model configuration.
- [ ] The challenge website’s required protocol has been identified from evidence.
- [ ] The value pasted into the challenge website matches the mapping in section 5.
- [ ] If the challenge runs in a browser on a different origin, CORS has been explicitly allowlisted and tested.
- [ ] Rate limiting verified per client (two different networks get independent budgets).

## 8. If the challenge is not custom REST

Stop at the integration boundary and provide the challenge’s request/response example. The next implementation should add **one** adapter—Responses-style, A2A, or MCP—around the existing core agent service, then add one contract test against the challenge client. Do not bend `/api/chat` into an undocumented imitation of another protocol.
