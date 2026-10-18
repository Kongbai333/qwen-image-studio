# Security

## Reporting

Please report suspected vulnerabilities privately to the repository owner.
Do not open a public issue that contains credentials, private images, proxy
addresses, or exploitable deployment details.

## Secrets

The application stores runtime state outside source control:

- `tokens.txt` and `proxies.txt`
- `.admin.json` and `.session.key`
- `admin_password.txt`
- license, gallery, job, material, and channel JSON files
- uploaded and generated images

Never commit those files. If a token or password was committed, revoke it
immediately and replace it; deleting the file from the latest commit is not
enough because Git history retains it.

## Deployment

- Put the service behind HTTPS and a trusted reverse proxy before exposing it
  to the internet.
- Keep `--no-auth` disabled outside local development.
- Use only Hugging Face accounts, proxies, and Spaces that you are authorized
  to use, and follow the provider's terms and rate limits.
- Back up the data volume, but treat backups as sensitive because they contain
  access keys, prompts, and uploaded images.
