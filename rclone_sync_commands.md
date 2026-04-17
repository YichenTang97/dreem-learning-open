# First time

```
rclone bisync local_dir r2:research-data/dreem/data --resync --compare size,modtime --conflict-resolve newer --create-empty-src-dirs --modify-window 2s --resilient --recover --retries 10 --low-level-retries 20 --timeout 1m -vP
```

# After that
```
rclone bisync local_dir r2:research-data/dreem/data --conflict-resolve newer --resilient --recover --max-lock 2m --modify-window 2s --compare size,modtime --create-empty-src-dirs --retries 10 --low-level-retries 20 --timeout 1m -vP
```
