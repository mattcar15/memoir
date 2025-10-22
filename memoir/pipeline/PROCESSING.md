The approach for the image pipeline is going to be the following.

We have a triage setup for each image. 
Each image we first see if it's substantivley different from a memory in the last 5 minutes. we do this using pHash.
if the image is substantivley different than we process it.
when we process it we are going to first down scale the image. We then use ocr to pull the text out of the image.
If the text is less than 100 chars we then assume it's primarily an image. In this case we then pass it to the ollama processing we currently use.
After this we should have a breakdown of what the user is doing right now. We then use cosine similarity on open memories to see if it should get combined into an existing memory or if it's a new one.

existing memories are memories that have had an addition in the last 5 minutes. After 5 minutes of inactivity we consider a memory closed out.

if it joins an existing memory then we don't need to retain the image in it's full resolution. we should downscale it. this is then stored.

Our data retention works in the following way:

Hot (24 h): optional low-fps HEVC/AV1 timeline (0.2–0.4 Mbps) then purge.

Warm (7 days): text + tiny thumbs + 1/min keyframes.

Cold (30+ days): only summaries + entities (no images).

