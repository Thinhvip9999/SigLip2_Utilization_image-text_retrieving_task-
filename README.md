# SigLip2_Utilization_image-text_retrieving_task

This repository presents the implementation and optimization of the SigLip2 model for large-scale image-text retrieval tasks. It is developed and executed on Server 5060 with GPU acceleration, using a massive dataset of over 400,000 high-resolution images. The project is part of my participation in the HCM_AI competition, aiming to build an efficient and scalable multimodal retrieval system that connects visual and textual representations effectively.

The SigLip2 architecture leverages contrastive learning to align image and text embeddings in a shared latent space, enabling accurate and fast retrieval across a wide range of multimodal queries. The system is optimized for large-scale experiments, supporting batch processing, checkpoint saving, and reproducible results.

## Key Features
- Implementation of SigLip2 for large-scale multimodal retrieval tasks  
- Runs on Server 5060 with full GPU acceleration  
- Dataset includes over 400K image-text pairs for training and evaluation  
- Optimized for embedding extraction and similarity search  
- Includes preprocessing, augmentation, and evaluation scripts  
- Designed for use in the HCM_AI competition environment  

## Technical Overview
This repository provides:
- Code for image and text feature extraction using SigLip2  
- Scripts for non-square image processing (see siglip2_process_with_non_squared_image.py)  
- Large-scale embedding generation and retrieval evaluation  
- Example workflow for fine-tuning and testing retrieval performance  
- Documentation in Guidance.md and the main README.md for reference  

## Objective
The primary objective of this project is to bridge the semantic gap between visual and textual information by leveraging large-scale multimodal data. Through efficient training and evaluation pipelines, the system provides a foundation for real-world applications such as image search, caption alignment, and AI-based semantic retrieval.

---

**Author:** Thinhip9999  
**Competition:** HCM_AI  
**Dataset Size:** >400K images  
