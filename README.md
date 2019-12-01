# Odstraňovanie šumu z obrázkov
Návrh projektu z neurónových sietí FIIT STU 2019/20

## Motivácia
V dnešnej dobe plnej vizuálnych médií sa kladie veľký dôraz na kvalitu obrazu, ktorá je však daná použitým zariadením na snímanie. Použitie technicky slabších fotoaparátov môže zapríčiniť, že výsledný obraz bude obsahovať artefakty, ako napríklad obrazový šum. Nepresnosti však môžu byť zapríčinené aj fyzickým znehodnotením filmu, či skenovaním fotografie. Prinavrátiť, či zvýšiť kvalitu takýmto zašumeným obrázkom vyžaduje veľa ľudského úsilia. Na základe zdrojov preto navrhujeme natrénovať konvolučnú sieť, ktorá bude odstraňovať rôzne typy šumu z fotografií.

## Súvisiaca práca
Na podobné úlohy sa v súčasnosti používajú predovšetkým konvolučné siete.
- [A Convolutional Neural Networks Denoising Approach for Salt and Pepper Noise](https://arxiv.org/ftp/arxiv/papers/1807/1807.08176.pdf)
- [Natural Image Noise Dataset](https://arxiv.org/abs/1906.00270)
- [Multi-level Wavelet-CNN for Image Restoration](https://arxiv.org/pdf/1805.07071v2.pdf)
- [Beyond a Gaussian Denoiser: Residual Learning of Deep CNN for Image Denoising](https://arxiv.org/pdf/1608.03981.pdf)
- [Abdelrahman Abdelhamed, Lin S., Brown M. S. "A High-Quality Denoising Dataset for Smartphone Cameras", IEEE Computer Vision and Pattern Recognition (CVPR), June 2018.](https://www.eecs.yorku.ca/~kamel/sidd/files/SIDD_CVPR_2018.pdf)

## Datasety
Existuje viacero datasetov obsahujúcich zašumené aj vyčistené verzie tých istých obrázkov. Pri niektorých je aj možnosť porovnania výsledkov s inými metódami.  
- [Smartphone Image Denoising Dataset](https://www.eecs.yorku.ca/~kamel/sidd/dataset.php)
- [Darmstadt Noise Dataset](https://noise.visinf.tu-darmstadt.de/)
- [Natural Image Noise Dataset](https://commons.wikimedia.org/wiki/Natural_Image_Noise_Dataset)

Taktiež je možnosť vygenerovať šum pomocou editoru obrázkov na vlastných fotografiách, a pokúšať sa o ich opravu. Pri tomto prístupe je však dôležité mať na pamäti použitie rôznych druhov generovaného šumu, aby bol náš model čo najviac univerzálny.

