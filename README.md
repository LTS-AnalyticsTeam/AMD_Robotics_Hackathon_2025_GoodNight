# AMD_Robotics_Hackathon_2025_GoodNight
## Title:
**AMD_RoboticHackathon2025-GoodNight**

## Team:
**LTS Robotics Team**
This team is made up of members from LTS, Inc.
- **Keito Sonehara**
- **Haruki Yoshida**
- **Kazuki Kubota**
- **Wonjun Choi**

## Summary:
This robot gently tucks a blanket over a plush toy or a person using its two arms.
It can place, adjust, and smooth the blanket to provide comfort and warmth on a cold night.
Through this project, we aim to bring a sense of kindness and emotional warmth to machines that are often perceived as mechanical and impersonal.
![alt text](./media/img/image.png)

## Submission Details

### 1. Mission Description
Our mission explores how robots can provide not only functional assistance but also emotional comfort in everyday life.  
While conventional home robots focus on efficiency and automation, our approach aims to introduce warmth, empathy, and gentleness into human–robot interaction.

This project tackles several real-world challenges:
- Providing a comforting presence for people who sleep alone  
- Maintaining a pleasant sleeping environment through gentle blanket adjustments  
- Supporting children, elderly individuals, or anyone who has difficulty handling bedding independently  

The goal is not merely to manipulate a blanket, but to demonstrate how robots can contribute to human well-being by performing delicate, caring actions.

### 2. Creativity
Our work introduces several novel aspects:
- A bi-manual coordination strategy using two SO-101 arms to lift, spread, and place a blanket with gentle and expressive motion  
- A design concept emphasizing emotional support rather than pure task performance  
- Motion profiles and policy tuning aimed at expressing softness, care, and non-intrusiveness  
- A GUI-based interaction system that allows users to trigger high-level commands such as “Good night” and “Good morning”

These components collectively highlight a new direction in home robotics: robots that feel emotionally considerate while remaining technically robust.

### 3. Technical Implementations

#### Teleoperation / Dataset Capture
To reliably manipulate a highly deformable blanket, we refined both the task setup and the way the arms are operated during teleoperation.  
After grasping, we carefully adjusted lift direction, arm posture, and motion speed so that the blanket would not slip or fall, avoiding unnecessary tension or twisting of the material.
We developed a custom bi-manual teleoperation environment where two SO-101 arms can be jointly controlled and recorded as a single synchronized dataset.  
Through repeated practice and refinement, we collected nearly 500 high-quality episodes.  
To improve data quality, we performed trimming, filtering, and refinement of teleoperation trajectories before training.

#### Training
We applied multiple rounds of hyperparameter tuning to improve policy stability and blanket manipulation accuracy.  
We experimented with different learning rates, action chunk sizes, and dataset cleaning strategies.  

#### Hardware Configuration
We built a dual-arm setup using two SO-101 units with synchronized control.  
This setup allows:  
- Simultaneous bi-manual teleoperation  
- Unified synchronized dataset recording  
- Paired policy inference for both arms  

#### GUI Application
We implemented an interactive GUI that enables high-level, user-friendly commands:
- Good night – the robot gently places a blanket  
- Good morning – the robot carefully pulls the blanket away
This interface reduces operational complexity and enhances usability for non-expert users.

### 4. Ease of Use
Our system is designed with practical deployment and extensibility in mind:
- The bi-manual framework can generalize to other dual-arm tasks, such as folding, organizing, or coordinated object manipulation  
- The teleoperation interface and dataset pipeline support rapid collection of new demonstrations for other domains  
- The GUI enables intuitive control, hiding low-level robot details and providing simple, understandable commands  
- The overall architecture is modular, allowing teams to adapt the approach to different environments, tasks, or robot platforms  

The result is a flexible, user-friendly system that balances creativity, technical rigor, and real-world applicability.


## How To:
### Delivery URL
- https://huggingface.co/collections/lt-s/amd-hackathon-2025
### Bi manual
For Bi Manual operation, please refer to the following document.
- [bi_so101_bimanual.md](mission2/docs/bi_so101_bimanual.md)

## Demo Video
The following video shows a successful blanket-tucking action on a doll.


https://github.com/user-attachments/assets/22517c3b-e972-49ee-a0ac-ce98cedb2a9b

[another videos](media/video)
