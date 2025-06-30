SES=ses_8552f9d36ffeT6R8ZcNpyyFGg9   # ← the folder you want
DIR=~/.local/share/opencode/project/Users-jacob-Documents-Sandbox-microbial_claude_matter/storage/session/message/$SES

for f in $(ls -1 "$DIR"/msg_*.json | sort); do
  jq -r '"\(.role):\n\(.content)\n"' "$f"
done | less -R
