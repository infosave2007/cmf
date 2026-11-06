# Dory turbo: a fixed ceiling while a long prompt is read, 05.11.2026

Goal: Dory kept 0.5 GB of device KV after a prompt and all its rows on the card while reading it. Rows past the first seal now
go to RAM or to files, as on Qwen3.8-27B. On branch `dory-spill`, over PR #85 `q27-turbo`.
Stand: RTX 5090, 32 GB, Vulkan, EPYC host, 186 GB of RAM, a disk with 4.2 GB/s write. MacBook Air M4, 24 GB, for Metal.
Scripts: `spill.sh`, `spill_final.sh`, `spill_metal.sh`, `passkey_prompt.py`, `vram_peak.sh`. Logs on the pod: `/root/q/logs/dory_{before,after,final}.log`.

## Where the card's memory went

- Each full slot grew by doubling to 65536 rows of K and V, with the old and the new buffer alive during the copy.
- `dory-qtrace` was allocated for `ARCH_MAX` from the first token: 66 MB per full layer, at a prompt of any length. That is the
  1 GB gone at 1000 tokens below.
- At the first seal the far states went up while the slots still held their rows.

## What changed

- wgpu: past the first seal a full slot holds one chunk. `dory_arch` and the new `dory_qarch` copy the chunk's K, V and Q there
  in the plan, with no break in the compute pass; at the chunk's end a fixed staging buffer takes all the slots at once, and the
  rows go to `ArchStore`. Queries are filtered on the host by the absolute position, one row in 8, so a prompt that does not start
  at a multiple of 8 gets the same rows as the shader would.
- Before the prompt's last 512 rows the choice of exact rows reads the stores, and `seal_archive` rebuilds the far field one
  layer at a time. Files are removed as a layer is done.
- `qtrace` is now sized for the first seal. A first seal frees a layer's slot as soon as its far state is up.
- Metal: seal at 8192 rows when a file has exact rows and `o1c` takes the far field, sealed chunks through `o1c::encode` and a new
  `dory_gate1`. Before, rows past 32768 went one by one and no row past the seal could be chosen.
- CPU: the same stores (`SlotStore`), one row's queries in 8 past the first seal. It was every row: 8 KB a row per full layer.
- A store that fails (no directory, a full disk) costs the exact rows and not the answer: the far field stays as first sealed, and
  a warning says so. Tests `a_spill_directory_that_cannot_be_made_leaves_the_far_field_working` and
  `a_prompt_past_the_seal_reads_the_same_from_ram_and_from_files`.

## Vulkan, peaks over a whole `bench --core`

MiB, as `nvidia-smi` gives them, the card's use before the run subtracted. `vram_peak.sh` samples every 0.5 s, so the old binary's
real peak can be higher.

| prompt | device, old | device, new | RAM, old | RAM, new | files: RAM | files: disk |
|---:|---:|---:|---:|---:|---:|---:|
| 1 000 | 4162 | 3154 | 204 | 204 | | |
| 12 000 | 5454 | 3849 | 1572 | 1460 | 1183 | 2087 |
| 32 000 | 7581 | 3849 | 4021 | 3707 | | |
| 64 000 | 11039 | 3849 | 7363 | 7336 | 1777 | 9863 |

Speed at 64 000, old / RAM / files: prefill 707.5 / 699.6 / 704.7 tok/s, first token 90.2 / 91.4 / 90.8 s, decode 126.2 / 127.1 / 128.3.
The sync per chunk costs about 1 %. At 1000 tokens the speed did not move: 1406 / 1405 prefill, 158.8 / 159.1 decode.
Final rebuild over 63 488 rows: 4.8 s, the choice 0.25 s. The first seal's 3.8 s are mostly a wait for the queue.

A row takes about 159 KB in files: K, V and one row's queries in 8 over the full layers.

## Quality

- Greedy answer of 120 tokens at 100 000 characters, `CMF_GPU_PROBE=0`: the old binary, RAM and files give the same text.
- `passkey2.py`, 100 000 characters: old 8/8, RAM 8/8, files 8/8. First token about 32 s in all three.
- `passkey_serve.py`, 250 000 characters, about 58 000 rows, files against the old binary: 2/4 and 2/4, the same trials missed with the
  same output, 79086 for 790869 and 12 for 621287. Each read through the server took 83 s both ways, and no file stayed.

After these runs two edits went in that a good run never reaches: the size of rows restored after a refused state, and a false warning
on the CPU. The numbers are from the binary before them.

## Metal, MacBook Air M4

Peak footprint without the mapped weights, one planted fact.

| prompt | old | RAM | files |
|---:|---:|---:|---:|
| 100 000 characters, 23 480 rows | 4.12 GB, 240 s | 3.27 GB, 187 s | 1.64 GB, 207 s |
| 250 000 characters, 58 216 rows | not run | 7.02 GB, 565 s | not run |

The fact came back in all of them. On unified memory RAM mode is no saving on a very long prompt, since the rows only move within the
same memory: there the files are what helps, and rows past 32768 are now chosen from. The 565 s were taken with a `cargo test` running
for part of the time.

## Traps

- macOS has no `setsid`: `setsid nohup X &` fails silently in a subshell, and the first Metal chain never started.
- `du` on APFS undercounts files being written: 1980 MB for what is about 3.6 GB. Use the numbers from Linux.
- A compile on the Mac during a measure is still a false number.
- A store check that ran on every chunk printed "a store failed" through the prompt's tail on the CPU. A check after a real run of the
  file found it: the tests with a tiny model do not read logs.

## Not done

- A real 4 GB card: 3849 MiB on the 5090 leaves very little there. 8192 exact rows take about 1 GB of it. A lower seal would give
  that back, and costs exactness and speed on prompts from 4096 to 8192 rows; not measured.
- A prompt past 65536 rows still chooses at `ARCH_MAX` and reads the rest row by row, as before.
- CPU past 58 000 rows: not run. Real file at 40 000 characters, 9280 rows, Mac: new in RAM, new in files and old all take 287 s to the
  first token and 2.6 GB. All three miss the planted fact, the old one with `1780148` for 469030, and all answer in the same 8 tokens, so
  this is not from the change. The difference in memory needs a much longer prompt.
