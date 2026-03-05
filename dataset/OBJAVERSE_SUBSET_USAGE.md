# Objaverse subset downloader

Scripts:
- `/root/cube/dataset/download_objaverse_subset.py`: core resumable downloader.
- `/root/cube/dataset/run_objaverse_subset_download.sh`: runs downloader after `source /etc/network_turbo`.
- `/root/cube/dataset/start_objaverse_subset_bg.sh`: starts downloader in background with `nohup`.

Data output:
- `/root/autodl-tmp/objaverse_subset/hf-objaverse-v1/glbs`: downloaded `.glb` files.
- `/root/autodl-tmp/objaverse_subset/manifests/pairs.jsonl`: model-text pairs.
- `/root/autodl-tmp/objaverse_subset/manifests/completed_uids.txt`: resume state.

Run in foreground:
```bash
bash /root/cube/dataset/run_objaverse_subset_download.sh --target-gb 38 --min-free-gb 8 --max-retries 8
```

Run in background:
```bash
bash /root/cube/dataset/start_objaverse_subset_bg.sh 38 8 8
```

Check progress:
```bash
tail -f /root/autodl-tmp/objaverse_subset/download.log
du -sh /root/autodl-tmp/objaverse_subset/hf-objaverse-v1/glbs
wc -l /root/autodl-tmp/objaverse_subset/manifests/completed_uids.txt
```

Resume later:
- Re-run either command above; it will skip completed UIDs automatically.
