# 05 · Deploy to Streamlit Community Cloud

Result: a public URL like `https://your-app-name.streamlit.app` that you put on your CV/LinkedIn.

## Before you start

- [ ] App runs locally with your real `artifacts/` (Guide 04 checklist passed)
- [ ] `model.joblib` is **under 100 MB**
- [ ] `requirements.txt` has pinned versions (Guide 03, Step 3)
- [ ] GitHub account + Streamlit Cloud account (sign in with GitHub)

---

## Step 1 — Create `.gitignore`

In the project root create a file named `.gitignore`:

```
.venv/
__pycache__/
*.pyc
.ipynb_checkpoints/
.DS_Store
```

⚠️ **Do NOT ignore `artifacts/`** — the deployed app needs those files.

## Step 2 — Create the GitHub repository

1. github.com → **New repository**
2. Name: `fraud-risk-app` · Visibility: **Public** (free Streamlit Cloud deploys from public repos easily;
   private repos need extra permission) · don't tick "add README" (you already have one)
3. Click **Create repository**

## Step 3 — Push your code

In the project folder:

```bash
git init
git add .
git commit -m "Fraud risk analyzer: Streamlit app + trained model"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/fraud-risk-app.git
git push -u origin main
```

Refresh the GitHub page — you should see `app/`, `artifacts/`, `docs/`, `requirements.txt`.

**Push rejected for a large file?** (`file exceeds GitHub's file size limit of 100 MB`) Go back to
Guide 03 and shrink the model (Option A/B/C). Removing the file in a new commit is *not* enough because
Git keeps history — run `git reset --soft HEAD~1`, fix the file, commit again.

## Step 4 — Deploy

1. Go to **share.streamlit.io** and sign in with GitHub (authorise access when asked).
2. Click **Create app** / **New app** → "Deploy a public app from GitHub".
3. Fill in:
   | Field | Value |
   |---|---|
   | Repository | `YOUR_USERNAME/fraud-risk-app` |
   | Branch | `main` |
   | Main file path | `app/app.py` |
   | App URL | choose a custom name, e.g. `fraud-risk-analyzer` |
4. Open **Advanced settings** and choose the **Python version** matching your training environment
   (e.g. 3.11 or 3.12).
5. Click **Deploy**. The first build takes a few minutes (installing packages). Watch the logs on the
   right — red text tells you what failed.

The Streamlit Cloud interface changes occasionally; if labels differ slightly, the docs at
docs.streamlit.io → "Deploy your app" are the source of truth.

## Step 5 — Verify the live app

Repeat the Guide 04 checklist on the live URL, **from your phone too**. Then:

- Add the URL to the top of your GitHub README, your CV, and LinkedIn.
- In the app's menu (⋮) check **Settings → Sharing** is public.

## Updating later

```bash
git add .
git commit -m "Describe your change"
git push
```
The live app redeploys automatically within a minute or so.

---

## Things interviewers will experience

| Behaviour | What to do |
|---|---|
| Free apps **go to sleep after inactivity** and show a "wake up" button; waking takes some seconds | Open the link yourself ~10 minutes before the interview; mention it in your CV line ("may take ~30 s to wake") |
| Several people use it at once | The cached model is shared; fine for a demo |
| Slow first prediction | Model loading happens once (cached) |

## Backup plan (always have one)

- Record a 60–90 second screen video of the app working.
- Keep it running locally (`streamlit run app/app.py`) on your laptop.
- Put 2–3 screenshots in your README.

## Optional: custom README for GitHub

Use the included `README.md` — fill in your live URL and name. A good README is the first thing a
reviewer reads.

## Security & privacy reminders

- Never commit passwords, API keys or tokens. If you add any secret (e.g. a Hugging Face token), use
  Streamlit Cloud **Settings → Secrets** and read it with `st.secrets["NAME"]`.
- Uploaded CSVs are processed in memory; still, tell users not to upload real card data. The app is a
  demo trained on public synthetic-looking data.
