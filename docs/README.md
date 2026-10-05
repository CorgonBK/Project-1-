# AI & Computer Vision for Protecting the Elderly

A complete GitHub Pages research tutorial on fall detection and safe aging-in-place using modern computer vision.

**Author:** Bryant Kha

## Contents

| Page | Description |
|------|-------------|
| `index.html` | Home & introduction |
| `problem.html` | Why falls matter |
| `how-it-works.html` | System pipeline |
| `techniques.html` | Algorithms & pose estimation |
| `systems.html` | Real systems (UAlberta, edge, mobile) |
| `challenges.html` | Privacy, false alarms, limitations |
| `future.html` | Research directions |
| `quiz.html` | Interactive self-check quiz |
| `bibliography.html` | Annotated bibliography (6 references) |
| `styles.css` | Shared styling |

## How to publish on GitHub Pages

1. Create a new repository (or use your existing project repo).
2. Upload **all** files and folders from this directory to the root of the repository
   (or into a `/docs` folder if you prefer).
3. Go to **Settings → Pages**.
4. Under “Source”, choose the branch (`main`) and folder (`/ (root)` or `/docs`).
5. Save. After a minute your site will be live at:
   `https://project-1-.github.io/REPO_NAME/`

## Adding the audio narrations

Each page contains an `<audio>` tag pointing to files such as `audio/home.mp3`.

You can generate these files for free:

- Use Windows Narrator + a recorder, or
- Use any free Text-to-Speech website (e.g., ttsmp3.com, Google Translate TTS, ElevenLabs free tier),
- Save the MP3s into the `audio/` folder with the exact names used in the HTML.

The narration scripts are included as HTML comments at the bottom of `index.html` (and can be expanded for the other pages).

## Adding images

Place your diagrams in the `images/` folder:

- `fall-statistics.png`
- `pipeline.png`
- `pose-landmarks.png`
- `system-comparison.png`

Even simple PowerPoint / draw.io exports are fine. Always credit the source in the figure caption if you use an external image.

## Testing checklist

- [ ] All navigation links work
- [ ] Quiz answers appear correctly
- [ ] Bibliography anchor links (`#ref1` … `#ref6`) jump to the right entry
- [ ] Site looks good on both desktop and phone
- [ ] Audio players appear (even if the MP3s are still missing)

Enjoy the tutorial!
