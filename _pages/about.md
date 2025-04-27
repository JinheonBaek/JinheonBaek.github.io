---
permalink: /
title: ""
excerpt: ""
author_profile: true
redirect_from: 
  - /about/
  - /about.html
---

{% if site.google_scholar_stats_use_cdn %}
{% assign gsDataBaseUrl = "https://cdn.jsdelivr.net/gh/" | append: site.repository | append: "@" %}
{% else %}
{% assign gsDataBaseUrl = "https://raw.githubusercontent.com/" | append: site.repository | append: "/" %}
{% endif %}
{% assign url = gsDataBaseUrl | append: "google-scholar-stats/gs_data_shieldsio.json" %}

<span class='anchor' id='about-me'></span>

---

**Jinheon Baek (백진헌)** (jinheon.baek \[at] kaist \[dot] ac \[dot] kr), and here is my [CV (Curriculum Vitae)](/assets/files/cv.pdf)

---

**I'm a Ph.D. student** in the Graduate School of AI at KAIST [(MLAI Lab)](https://www.mlai-kaist.com/), where I am fortunate to be advised by Prof. [Sung Ju Hwang](http://www.sungjuhwang.com/). Before that, I received an M.S. degree in Artificial Intelligence at KAIST in 2022. Prior to studying at KAIST, I earned my B.S. degree in Computer Science and Engineering at Korea University in 2020, where I studied machine learning under the guidance of Prof. [Jaewoo Kang](https://dmis.korea.ac.kr/jaewoo-kang-p-i). 

Previously, I worked as a research intern at **Google DeepMind (Gemini)** in 2024, as a research intern at **IBM Research** in 2024, as a research intern at **Microsoft Research** in 2023, and as an applied scientist II intern at **Amazon (Alexa AI)** in 2022, collaborating with wonderful mentors.

**My primary research interest** lies in the area of machine learning for language, knowledge, and their intersections at scale. Previous work includes modeling interconnected data structures (texts and graphs), and retrieving them to augment language models for practical natural language applications. You can refer to my [Research Statement: Augmenting Large Language Models with External Knowledge](/assets/files/rs.pdf), if interested in.

<!-- # 🔥 News
- *2022.02*: &nbsp;🎉🎉 Lorem ipsum dolor sit amet, consectetur adipiscing elit. Vivamus ornare aliquet ipsum, ac tempus justo dapibus sit amet. 
- *2022.02*: &nbsp;🎉🎉 Lorem ipsum dolor sit amet, consectetur adipiscing elit. Vivamus ornare aliquet ipsum, ac tempus justo dapibus sit amet.  -->

# 📝 Publications 

<!-- <div class='paper-box'><div class='paper-box-image'><div><div class="badge">CVPR 2016</div><img src='images/500x300.png' alt="sym" width="100%"></div></div>
<div class='paper-box-text' markdown="1">

[Deep Residual Learning for Image Recognition](https://openaccess.thecvf.com/content_cvpr_2016/papers/He_Deep_Residual_Learning_CVPR_2016_paper.pdf)

**Kaiming He**, Xiangyu Zhang, Shaoqing Ren, Jian Sun

[**Project**](https://scholar.google.com/citations?view_op=view_citation&hl=zh-CN&user=DhtAFkwAAAAJ&citation_for_view=DhtAFkwAAAAJ:ALROH1vI_8AC) <strong><span class='show_paper_citations' data='DhtAFkwAAAAJ:ALROH1vI_8AC'></span></strong>
- Lorem ipsum dolor sit amet, consectetur adipiscing elit. Vivamus ornare aliquet ipsum, ac tempus justo dapibus sit amet. 
</div>
</div> -->

(\* denotes the equal contribution)

- [Revisiting In-Context Learning with Long Context Language Models](https://arxiv.org/abs/2412.16926) \\
**Jinheon Baek**, Sun Jae Lee, Prakhar Gupta, Geunseob Oh, Siddharth Dalmia, and Prateek Kolhar \\
arXiv preprint

- [Knowledge Base Construction for Knowledge-Augmented Text-to-SQL]() \\
**Jinheon Baek**, Horst Samulowitz, Oktie Hassanzadeh, Dharmashankar Subramanian, Sola Shirai, Alfio Gliozzo, and Debarun Bhattacharjya \\
arXiv preprint

- [VideoRAG: Retrieval-Augmented Generation over Video Corpus](https://arxiv.org/abs/2501.05874) \\
Soyeong Jeong\*, Kangsan Kim\*, **Jinheon Baek**\*, and Sung Ju Hwang \\
arXiv preprint

- [Unified Multi-Modal Interleaved Document Representation for Retrieval](https://arxiv.org/abs/2410.02729) \\
Jaewoo Lee\*, Joonho Ko\*, **Jinheon Baek**\*, Soyeong Jeong, and Sung Ju Hwang \\
arXiv preprint

- [Efficient Long Context Language Model Retrieval with Compression](https://arxiv.org/abs/2412.18232) \\
Minju Seo, **Jinheon Baek**, Seongyun Lee, and Sung Ju Hwang \\
arXiv preprint

- [Real-time Verification and Refinement of Language Model Text Generation](https://arxiv.org/abs/2501.07824) \\
Joonho Ko, **Jinheon Baek**, and Sung Ju Hwang \\
arXiv preprint

- [Database-Augmented Query Representation for Information Retrieval](https://arxiv.org/abs/2406.16013) \\
Soyeong Jeong, **Jinheon Baek**, Sukmin Cho, Sung Ju Hwang, and Jong C. Park \\
arXiv preprint

- [ResearchAgent: Iterative Research Idea Generation over Scientific Literature with Large Language Models](https://arxiv.org/abs/2404.07738) \\
**Jinheon Baek**, Sujay Kumar Jauhar, Silviu Cucerzan, and Sung Ju Hwang \\
Conference of the Nations of the Americas Chapter of the Association for Computational Linguistics **(NAACL), 2025**

- [The BiGGen Bench: A Principled Benchmark for Fine-grained Evaluation of Language Models with Language Models](https://arxiv.org/abs/2406.05761) \\
Seungone Kim, Juyoung Suk, Ji Yong Cho, Shayne Longpre, Chaeeun Kim, Dongkeun Yoon, Guijin Son, Yejin Cho, Sheikh Shafayat, **Jinheon Baek**, ..., Graham Neubig, Moontae Lee, Kyungjae Lee, and Minjoon Seo \\
Conference of the Nations of the Americas Chapter of the Association for Computational Linguistics **(NAACL), 2025**

- [CVQA: Culturally-diverse Multilingual Visual Question Answering Benchmark](https://arxiv.org/abs/2406.05967) \\
David Romero, Chenyang Lyu, Haryo Akbarianto Wibowo, Teresa Lynn, Injy Hamed, Aditya Nanda Kishore, ..., **Jinheon Baek**, ..., Soyeong Jeong, ..., Thamar Solorio, and Alham Fikri Aji \\
Conference on Neural Information Processing Systems **(NeurIPS), 2024** **(Oral Presentation)**

- [Retrieval-Augmented Data Augmentation for Low-Resource Domain Tasks](https://arxiv.org/abs/2402.13482) \\
Minju Seo\*, **Jinheon Baek**\*, James Thorne, and Sung Ju Hwang \\
Adaptive Foundation Models Workshop at NeurIPS **(AFM @ NeurIPS), 2024**

- [An Empirical Study of Multilingual Reasoning Distillation for Question Answering](https://aclanthology.org/2024.emnlp-main.442/) \\
Patomporn Payoungkhamdee, Peerat Limkonchotiwat, **Jinheon Baek**, Potsawee Manakul, Can Udomcharoenchaikit, Ekapol Chuangsuwanich, and Sarana Nutanong \\
Empirical Methods in Natural Language Processing **(EMNLP), 2024**

- [Rethinking Code Refinement: Learning to Judge Code Efficiency](https://arxiv.org/abs/2410.22375) \\
Minju Seo, **Jinheon Baek**, and Sung Ju Hwang \\
Findings of Empirical Methods in Natural Language Processing **(Findings of EMNLP), 2024**

- [Adaptive-RAG: Learning to Adapt Retrieval-Augmented Large Language Models through Question Complexity](https://arxiv.org/abs/2403.14403) \\
Soyeong Jeong, **Jinheon Baek**, Sukmin Cho, Sung Ju Hwang, and Jong C. Park \\
Conference of the North American Chapter of the Association for Computational Linguistics **(NAACL), 2024**

- [Knowledge-Augmented Large Language Models for Personalized Contextual Query Suggestion](https://arxiv.org/abs/2311.06318) \\
**Jinheon Baek**, Nirupama Chandrasekaran, Silviu Cucerzan, Allen herring, and Sujay Kumar Jauhar \\
The Web Conference **(WWW), 2024**

- [Knowledge-Augmented Language Model Verification](https://arxiv.org/abs/2310.12836) \\
**Jinheon Baek**, Soyeong Jeong, Minki Kang, Jong C. Park, and Sung Ju Hwang \\
Empirical Methods in Natural Language Processing **(EMNLP), 2023**

- [Test-Time Self-Adaptive Small Language Models for Question Answering](https://arxiv.org/abs/2310.13307) \\
Soyeong Jeong, **Jinheon Baek**, Sukmin Cho, Sung Ju Hwang, and Jong C. Park \\
Findings of Empirical Methods in Natural Language Processing **(Findings of EMNLP), 2023**

- [Knowledge-Augmented Reasoning Distillation for Small Language Models in Knowledge-Intensive Tasks](https://arxiv.org/abs/2305.18395) \\
Minki Kang, Seanie Lee, **Jinheon Baek**, Kenji Kawaguchi, and Sung Ju Hwang \\
Conference on Neural Information Processing Systems **(NeurIPS), 2023**

- [Direct Fact Retrieval from Knowledge Graphs without Entity Linking](https://arxiv.org/abs/2305.12416) \\
**Jinheon Baek**, Alham Fikri Aji, Jens Lehmann, and Sung Ju Hwang \\
Annual Meeting of the Association for Computational Linguistics **(ACL), 2023**

- [Phrase Retrieval for Open-Domain Conversational Question Answering with Conversational Dependency Modeling via Contrastive Learning](https://arxiv.org/abs/2306.04293) \\
Soyeong Jeong, **Jinheon Baek**, Sung Ju Hwang, and Jong C. Park \\
Findings of the Association for Computational Linguistics **(Findings of ACL), 2023**

- [Knowledge-Augmented Language Model Prompting for Zero-Shot Knowledge Graph Question Answering](https://arxiv.org/abs/2306.04136) \\
**Jinheon Baek**, Alham Fikri Aji, and Amir Saffari \\
Natural Language Reasoning and Structured Explanations Workshop at ACL **(NLRSE @ ACL), 2023** **(Best Paper)** \\
Matching from Unstructured and Structured Data Workshop at ACL **(MATCHING @ ACL), 2023** **(Oral Presentation)**

- [Personalized Subgraph Federated Learning](https://arxiv.org/abs/2206.10206) \\
**Jinheon Baek**\*, Wonyong Jeong*, Jiongdao Jin, Jaehong Yoon, and Sung Ju Hwang \\
International Conference on Machine Learning **(ICML), 2023**

- [Realistic Conversational Question Answering with Answer Selection based on Calibrated Confidence and Uncertainty Measurement](https://arxiv.org/abs/2302.05137) \\
Soyeong Jeong, **Jinheon Baek**, Sung Ju Hwang, and Jong C. Park \\
Conference of the European Chapter of the Association for Computational Linguistics **(EACL), 2023**

- [Graph Self-supervised Learning with Accurate Discrepancy Learning](https://arxiv.org/abs/2202.02989) \\
Dongki Kim*, **Jinheon Baek**\*, and Sung Ju Hwang \\
Conference on Neural Information Processing Systems **(NeurIPS), 2022**

- [Object Detection in Aerial Images with Uncertainty-Aware Graph Network](https://arxiv.org/abs/2208.10781) \\
Jongha Kim, **Jinheon Baek**, and Sung Ju Hwang \\
Visual Object-oriented Learning meets Interaction Workshop at ECCV **(VOLI Workshop @ ECCV), 2022** 

- [Knowledge Graph-Augmented Language Models for Knowledge-Grounded Dialogue Generation](https://arxiv.org/abs/2305.18846) \\
Minki Kang\*, Jin Myung Kwak\*, **Jinheon Baek**\*, and Sung Ju Hwang \\
Knowledge Retrieval and Language Models Workshop at ICML **(KRLM Workshop @ ICML), 2022** 

- [KALA: Knowledge-Augmented Language Model Adaptation](https://arxiv.org/abs/2204.10555) \\
Minki Kang\*, **Jinheon Baek**\*, and Sung Ju Hwang \\
Annual Conference of the North American Chapter of the Association for Computational Linguistics **(NAACL), 2022** **(Oral Presentation)**

- [Augmenting Document Representations for Dense Retrieval with Interpolation and Perturbation](https://arxiv.org/abs/2203.07735) \\
Soyeong Jeong, **Jinheon Baek**, Sukmin Cho, Sung Ju Hwang, and Jong C. Park \\
Annual Meeting of the Association for Computational Linguistics **(ACL), 2022** **(Oral Presentation)** 

- [Toward Accurate Learning of Graph Representations](/assets/files/master_thesis.pdf) \\
**Jinheon Baek** \\
**Master’s Thesis**, KAIST, 2022

- [Edge Representation Learning with Hypergraphs](https://arxiv.org/abs/2106.15845) \\
Jaehyeong Jo\*, **Jinheon Baek**\*, Seul Lee\*, Dongki Kim, Minki Kang, and Sung Ju Hwang \\
Conference on Neural Information Processing Systems **(NeurIPS), 2021** 

- [Task-Adaptive Neural Network Retrieval with Meta-Contrastive Learning](https://arxiv.org/abs/2103.01495) \\
Wonyong Jeong\*, Hayeon Lee\*, Gun Park\*, Eunyoung Hyung, **Jinheon Baek**, and Sung Ju Hwang \\
Conference on Neural Information Processing Systems **(NeurIPS), 2021** **(Spotlight Presentation)**

- [Unsupervised Document Expansion for Information Retrieval with Stochastic Text Generation](https://arxiv.org/abs/2105.00666) \\
Soyeong Jeong, **Jinheon Baek**, ChaeHun Park, and Jong C. Park \\
Scholarly Document Processing Workshop at NAACL **(SDP Workshop @ NAACL), 2021** **(Oral Presentation)** 

- [Accurate Learning of Graph Representations with Graph Multiset Pooling](https://arxiv.org/abs/2102.11533) \\
**Jinheon Baek***, MinKi Kang\*, and Sung Ju Hwang \\
International Conference on Learning Representations **(ICLR), 2021** 

- [Exploring The Spatial Reasoning Ability of Neural Models in Human IQ Tests](https://www.sciencedirect.com/science/article/pii/S089360802100068X#!) \\
Hyunjae Kim\*, Yookyung Koh\*, **Jinheon Baek**, and Jaewoo Kang \\
**Neural Networks, 2021** 

- [Learning to Extrapolate Knowledge: Transductive Few-shot Out-of-Graph Link Prediction](https://arxiv.org/abs/2006.06648) \\
**Jinheon Baek**, Dong Bok Lee, and Sung Ju Hwang \\
Conference on Neural Information Processing Systems **(NeurIPS), 2020**

# 🎖 Honors and Awards
- Awarded the Presidential Science Scholarship for Graduate Study, 2024-2026
- Awarded the Travel Grant from KAIST-Google Partnership Program for WWW 2024
- Received the Best Poster Presentation Award at Samsung AI Forum 2023
- Received the Best Paper Award at NLRSE Workshop in ACL 2023
- Awarded the ICML Travel Grant for ICML 2023
- Awarded the Google Travel Grant for NeurIPS 2022
- Selected as One of the Top Reviewers (Top 10%) of NeurIPS 2022
- Selected as One of the Highlighted Reviewers (Top 10%) of ICLR 2022
- Selected as One of the Best Reviewers (Top 10%) of ICML 2021
- Received the Best Paper Award at CKAIA 2020
- Awarded the Samsung Dream Scholarship, 2016-2020
- Received the First Prize in the Graduation Project Competition at Korea University, 2019
- Received the Academic Excellence Award (highest GPA) at Korea University, 2019
- Received the Second Prize for Excellence in the Microsoft Student Partners Activities, 2018
- Nominated as the Representative of Korean for Excellence in Microsoft Student Partners Activities, 2017

<!-- # 📖 Educations
- *2019.06 - 2022.04 (now)*, Lorem ipsum dolor sit amet, consectetur adipiscing elit. Vivamus ornare aliquet ipsum, ac tempus justo dapibus sit amet. 
- *2015.09 - 2019.06*, Lorem ipsum dolor sit amet, consectetur adipiscing elit. Vivamus ornare aliquet ipsum, ac tempus justo dapibus sit amet. 

# 💬 Invited Talks
- *2021.06*, Lorem ipsum dolor sit amet, consectetur adipiscing elit. Vivamus ornare aliquet ipsum, ac tempus justo dapibus sit amet. 
- *2021.03*, Lorem ipsum dolor sit amet, consectetur adipiscing elit. Vivamus ornare aliquet ipsum, ac tempus justo dapibus sit amet.  \| [\[video\]](https://github.com/)

# 💻 Internships
- *2019.05 - 2020.02*, [Lorem](https://github.com/), China. -->