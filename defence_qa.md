# 🎓 External Defence — Complete Q&A Guide
**Project**: Deepfake Detection System (Multi-Model Ensemble)
**Students**: Tehzeeb Ul Hassan & Muhammad Khubaib Khalid
**Supervisor**: Mr. Adeel Mustafa
**Institute**: Institute for Art and Culture (IAC), Lahore

---

## 🔰 SECTION 1 — Project Basics

---

**Q1. Apna project explain karein?**

Hamne ek AI-powered Deepfake Detection System banaya hai jo images aur videos mein manipulated content detect karta hai. System 3 alag AI models (ViT-Large/CLIP, EfficientNet-B5, Xception71) ka ensemble use karta hai. User image ya video upload kare — system batata hai Real hai ya Fake, confidence percentage ke saath aur visual heatmap bhi dikhata hai.

---

**Q2. Yeh project kyun banaya?**

Generative AI (Midjourney, Stable Diffusion, Google Veo, Sora) itni advanced ho gayi hai ke fake images aur videos bilkul real lagte hain. Isse political manipulation, financial fraud, aur misinformation phail sakti hai. Koi aam user verify nahi kar sakta tha. Hamare project ne free, local, privacy-safe detector banaya.

---

**Q3. Deepfake kya hota hai?**

Deepfake ek AI technique hai jisme:
- Kisi ka chehra doosre par lagaya jata hai (Face Swap)
- Ya puri image/video AI se generate ki jati hai (Midjourney, DALL-E)
- Ya chehra ka expression manipulate hota hai (Face Reenactment)

"Deep" = Deep Learning. "Fake" = Fake content.

---

**Q4. Aapka system kya detect kar sakta hai?**

1. AI-Generated Images (Midjourney, Stable Diffusion, Gemini)
2. Face-Swap Deepfakes (DeepFaceLab, FaceSwap)
3. AI-Generated Videos (Google Veo, Sora, Kling)
4. GAN-generated faces (StyleGAN, ProGAN)

---

## 🤖 SECTION 2 — Models & Architecture

---

**Q5. Kitne models use kiye aur kyun?**

3 active models + 1 passive (weight=0):
1. ViT-Large/CLIP — Global AI pattern detector
2. EfficientNet-B5 — Face spatial consistency detector
3. Xception71 — Pixel-level forensic detector
4. ViT-Small — weight=0 (reliable nahi tha, exclude kiya)

Kyun 3? Ek model ki weakness doosra cover karta hai — ensemble se accuracy badh jati hai.

---

**Q6. Har model ka kaam alag kyun hai?**

Kyunki har model alag nazar se image dekhta hai:

| Model | Kya Dhundta Hai |
|-------|----------------|
| ViT-Large/CLIP | Poori image ka global AI texture pattern |
| EfficientNet-B5 | Face ka spatial balance aur blending |
| Xception71 | Pixel-level camera noise aur JPEG artifacts |

Jaise crime scene par 3 alag experts — doctor, fingerprint expert, behavior analyst. Teeno mil kar sahi conclusion nikaalte hain.

---

**Q7. ViT-Large/CLIP kyun choose kiya?**

ViT-Large/CLIP ko OpenAI ne billions of internet images par train kiya. Yeh model:
- Poori image ko ek saath "attention" se dekhta hai
- AI-generated images ke global texture patterns immediately pakadta hai
- Unseen deepfake generators par bhi kaam karta hai
- DFDC dataset par 87.2%, Celeb-DF par 96.6% accuracy

---

**Q8. EfficientNet-B5 kyun choose kiya?**

EfficientNet-B5 Google ka model hai jo compound scaling use karta hai — depth, width, resolution teeno ek saath scale karta hai:
- Face-swap mein blending boundary pakadta hai
- Chehre ke proportions ki spatial inconsistency detect karta hai
- 456x456 native resolution — zyada detail dekhta hai

---

**Q9. Xception kyun use kiya?**

Xception (Extreme Inception) Google ka model hai jo Depthwise Separable Convolutions use karta hai:
- Real camera ka natural sensor noise (PRNU) detect karta hai
- AI images mein yeh natural noise nahi hoti — Xception pakad leta hai
- Pixel-level blending seams aur texture mismatches detect karta hai
- Hamare system mein sabse honest model — real images ko correctly Real kehta hai

---

**Q10. CNN aur Vision Transformer (ViT) mein kya farq hai?**

| Feature | CNN | ViT |
|---------|-----|-----|
| Kaise dekhta hai | Local patches, ek ek karke | Poori image, globally |
| Mechanism | Convolution filters | Self-Attention |
| Best for | Texture, edge, local artifacts | Global pattern, context |
| Example | Xception, EfficientNet | ViT-Large/CLIP |

CNN = Microscope (local detail), ViT = Telescope (big picture)

---

**Q11. Ensemble kya hota hai aur kyun use kiya?**

Ensemble matlab multiple models ko mil kar ek final decision lena.

Hamara method: Weighted Soft Voting
- Har model apna probability score deta hai (0% to 100%)
- Har model ko alag weight diya gaya hai
- Weighted average se final score nikalta hai
- Agar score > threshold = FAKE, warna REAL

Kyun? ViT-Large strong hai AI images par, Xception strong hai real pakadne mein — mil kar dono ki weaknesses cover hoti hain.

---

**Q12. Weights kaise decide kiye?**

Weights empirically calibrate kiye gaye — real test data par trial karke:

Image Weights:
- ViT-Large/CLIP: 2.0 (50% voting share)
- EfficientNet-B5: 1.0 (25%)
- Xception71: 1.0 (25%)

Video Weights:
- ViT-Large/CLIP: 1.5 (37.5%)
- Xception71: 1.5 (37.5%) — video mein frame artifacts zyada pakadta hai
- EfficientNet-B5: 1.0 (25%)

---

**Q13. Threshold kya hota hai aur aapne kya rakha?**

Threshold wo cutoff point hai jis se upar score ho to FAKE, neeche to REAL.

- Image threshold: 0.38 (38%)
- Video threshold: 0.52 (52%)

Image threshold low (0.38) isliye taake moderate fakes bhi pakad sakein. Video threshold zyada (0.52) isliye kyunki 40 frames average kiye jate hain — zyada reliable score hota hai.

---

**Q14. Aapne models khud train kiye ya pre-trained use kiye?**

Pre-trained models use kiye. Training ke liye millions of images, weeks of GPU time chahiye hoti hai. Hamare project ka scope model training nahi tha.

Hamare actual contribution:
1. Teeno models ko ek unified pipeline mein integrate karna
2. Weighted ensemble system design karna
3. Thresholds empirically calibrate karna
4. Pakistani images par test aur fine-tune karna
5. Complete web application banana

---

**Q15. MTCNN kya hai aur kyun use kiya?**

MTCNN (Multi-task Cascaded Convolutional Networks) ek face detection model hai.

Pehle MTCNN face dhundta hai, phir sirf face crop ko models ko deta hai. Kyun?
- Models chehron par train hue hain — background se confuse nahi hone chahiye
- Face crop dene se accuracy badh jati hai
- 95% confidence required — agar face nahi mila to poori image use hoti hai

---

## 📊 SECTION 3 — Results & Performance

---

**Q16. Aapke system ki accuracy kitni hai?**

- Images: ~80-93% (test set ke hisaab se)
- Videos: ~89.6% (29 videos benchmark)
- Fake detection rate: ~100% (zero false negatives on our test set)

---

**Q17. False Positive aur False Negative kya hote hain?**

| Term | Matlab | Status |
|------|--------|--------|
| True Positive | Fake ko Fake bola | Correct |
| True Negative | Real ko Real bola | Correct |
| False Positive | Real ko Fake bola | Studio photos par hota hai |
| False Negative | Fake ko Real bola | Dangerous! Minimize kiya |

Hamare liye False Negative sabse dangerous tha — isliye 100% fake recall priority thi.

---

**Q18. Kuch images galat detect kyun hoti hain?**

Case 1 — Studio/Professional Photos (False Positive):
- Professional lighting flat hoti hai → AI generated jaisi lagti hai
- Edited skin → AI smoothing jaisi lagti hai
- ViT-Large confuse ho jata hai

Case 2 — High Quality Fakes (False Negative):
- Kuch fakes itne perfect hote hain ke teeno models low score dete hain
- Training data mein wohi fake type nahi thi
- Sirf retraining fix kar sakti hai

---

**Q19. Aapne 100% fake detection kaise achieve kiya?**

- Threshold 0.38 rakha — borderline fakes bhi pakad sakein
- ViT-Large ko 2x weight — AI fakes par most reliable
- Weighted soft voting — ek model ki low score poore result ko nahi bigaadti
- Multiple benchmarks par test karte rahe aur weights tune kiye

---

## 💻 SECTION 4 — Technical Implementation

---

**Q20. Backend technology kya use ki aur kyun?**

Python + Flask
- Python: PyTorch, timm, MTCNN ka best support
- Flask: Lightweight REST API — Django overkill hota

---

**Q21. Frontend technology kya use ki?**

React.js + Vite + Vanilla CSS
- React: Component-based UI — Scanner, History, Results alag components
- Vite: Fast build tool
- useState, useEffect — real-time animations ke liye

---

**Q22. Database kya use ki?**

SQLite (data/history.db)
- Zero setup — koi server nahi, sirf ek file
- Python built-in support
- WAL mode enable kiya — simultaneous read/write safe

---

**Q23. Video processing kaise hoti hai?**

1. Video upload hoti hai
2. MTCNN se faces detect hote hain
3. Video se 40 frames extract hote hain (5 FPS)
4. Har frame teeno models se analyze hota hai
5. Sab frames ke scores ka weighted average liya jata hai
6. Final score se FAKE/REAL decision

---

**Q24. Grad-CAM kya hai?**

Grad-CAM (Gradient-weighted Class Activation Mapping) ek explainability technique hai.

Dikhata hai ke model ne image ke kaunse hisse dekh kar decision kiya:
- Red area = Model ne yahan zyada suspicion dekha
- Blue area = Kam concern

Xception71 ke conv layers par apply hota hai — CNN layers par best kaam karta hai.

---

**Q25. Security measures kya hain?**

1. Magic Bytes Validation — File ka actual type check
2. Size Limits — Image max 10MB, Video max 100MB
3. Session ID — Har user ka alag folder
4. Path Traversal Block — ".." navigation prevent

---

**Q26. User ka data kahan jata hai?**

100% Local — koi data cloud par nahi jata:
- Files: Local uploads/ folder mein (temporary)
- History: Local SQLite file mein
- Koi API call bahar nahi jati

---

## 🔬 SECTION 5 — Deep Technical Questions

---

**Q27. Depthwise Separable Convolution kya hoti hai?**

Normal convolution: Spatial + Channel information ek saath process karta hai

Depthwise Separable (Xception mein):
- Step 1 (Depthwise): Har channel ko alag alag spatial filter se process karo
- Step 2 (Pointwise): 1x1 convolution se channels combine karo

Faida: Kam parameters, zyada efficient, zyada detail capture

---

**Q28. Self-Attention mechanism kya hai (ViT mein)?**

ViT image ko patches mein todta hai. Phir har patch doosre har patch se relationship dhundta hai:
- Aankhon wala patch socha: "Meri skin texture bilkul alag hai"
- Naak wala patch reply kiya: "Haan, mera bhi boundary unnatural hai"

Yeh global context CNN se behtar pakadta hai. AI images mein global inconsistency hoti hai jo ViT pakad leta hai.

---

**Q29. CLIP model kya hota hai?**

CLIP (Contrastive Language-Image Pretraining) — OpenAI ka model:
- Billions of (image, text) pairs par train hua
- Image aur text ko same embedding space mein lata hai
- "AI generated portrait" vs "real photo" deeply samajhta hai
- Hamare ViT-Large is CLIP pretraining se initialize hua — isliye generalization acha hai

---

**Q30. Native Resolution kyun important hai?**

Har model ek specific resolution par train hua:
- EfficientNet-B5: 456x456 pixels
- Xception71: 299x299 pixels
- ViT: 224x224 pixels

Agar galat resolution dein to model ke trained filters galat scale par apply hote hain — accuracy drop hoti hai.

Hamare system mein: Har model ko uski native resolution par input dete hain.

---

## ❓ SECTION 6 — Challenging Questions

---

**Q31. Aapka system fail kab hoga?**

1. Extreme quality fakes — perfect AI face
2. Pakistani studio photos — false positive
3. Very compressed images — camera artifacts destroy ho jaate hain
4. Partial face — MTCNN detect na kar sake
5. New unseen deepfake type — training mein nahi thi

Yeh known limitations hain, future work mein address ki ja sakti hain.

---

**Q32. Koi existing tool kyun nahi use kiya?**

Existing tools (Sensity, Microsoft Video Authenticator) mein problems:
- Cloud-based — privacy risk
- Paid/expensive
- Pakistani content par tested nahi
- Explainability (heatmap) nahi

Hamare system mein: Free, local, explainable, Pakistani content par tested.

---

**Q33. Model bias kya hota hai? Kya aapke system mein hai?**

Model bias = training data ki limitations system mein aati hain.

Hamare system mein:
- Models mostly Western faces par train hue — Pakistani faces par kabhi kabhi false positive
- Studio photos ke liye false alarm — training mein aise real photos kam the

Yeh valid limitation hai jo thesis mein honestly document ki gayi hai.

---

**Q34. Threshold 0.38 kyun rakha 0.5 kyun nahi?**

Test data par:
- Moderate fakes (39% score) 0.5 threshold par miss hote the
- 0.38 par yeh pakad lijiye bina new false alarms ke
- Real images ke scores 27% se neeche the — safe

Threshold empirically calibrate kiya — real test data par validate.

---

**Q35. Accuracy improve karne ke liye kya karo ge?**

1. Pakistani studio real photos training data mein add karna
2. Audio deepfake detection — voice cloning
3. Real-time stream detection
4. Model fine-tuning on Pakistani faces
5. Newer architectures — EVA-CLIP, DINOv2

---

## ⚡ SECTION 7 — Quick Reference

| Question | Short Answer |
|----------|-------------|
| Kitne models? | 3 active (ViT-Large, EfficientNet-B5, Xception71) |
| Training dataset? | FF++, DFDC, Celeb-DF, CommunityForensics |
| Image Accuracy? | ~80-93% |
| Video Accuracy? | ~89.6% |
| Fake Detection Rate? | ~100% |
| Backend? | Python + Flask |
| Frontend? | React + Vite + CSS |
| Database? | SQLite |
| Local ya Cloud? | 100% Local |
| Grad-CAM? | Xception71 conv layer par |
| Image Threshold? | 0.38 |
| Video Threshold? | 0.52 |
| Face Detection? | MTCNN (95% confidence) |
| Video Frames? | Max 40 frames @ 5 FPS |
| ViT-Small weight? | 0.0 (excluded) |

---

*"Hamare system ka core strength: 3 bilkul alag architectures mil kar ek decision lete hain — jisse kisi ek model ki limitation poore result ko affect nahi karti."*
