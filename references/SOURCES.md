# Sources and claim mapping

## 1. User-provided primary hardware sources

| Stored file | Use in the audit |
|---|---|
| [ROMO-B complete compressed manual](../source_files/hardware/ROMO-B_manual_complete_compressed.pdf) | geometry, steering limits, control modes, interface and feedback |
| [Livox Mid-360 guide](../source_files/hardware/Livox_Mid-360_quick_start_guide_ko.pdf) | KALO-H perception hardware context; not used to infer slip |
| [OpenArm drive commands](../source_files/hardware/OpenArm_drive_commands_jeongmin.pdf) | executable arm-command context |
| [OpenArm drive summary](../source_files/hardware/OpenArm_drive_summary_janghyun.pdf) | arm-operation context |
| [OpenArm ClassicCAN manual](../source_files/hardware/OpenArm_v1_ClassicCAN_ExoArm_command_manual.pdf) | arm interface context |
| [OpenArm command poster](../source_files/hardware/OpenArm_v1_ExoArm_command_poster.png) | arm-command reference |
| [OpenArm USD](../source_files/simulation/openarm_test.usd) | simulation asset context |
| [OpenArm USD v2](../source_files/simulation/openarm_test_v2.usd) | v2 simulation asset context |

Original filenames, local paths, byte sizes and SHA-256 values are recorded in
`source_manifest.tsv` and `SHA256SUMS`.

## 2. Direct novelty collisions

1. Chung and Velinsky, “Modeling and Control of a Mobile Manipulator,” *Robotica*, 1998.
   DOI: [10.1017/S0263574798000873](https://doi.org/10.1017/S0263574798000873)
   Relevant claim: mobile-manipulator dynamics with tire slip and tracking degradation/control.

2. Wang et al., “Dynamics Modeling of a Mobile Manipulator for Wheel Slip Avoidance,” IEEE ROBIO, 2011.
   DOI: [10.1109/ROBIO.2011.6181521](https://doi.org/10.1109/ROBIO.2011.6181521)
   Relevant claim: manipulator configuration changes wheel vertical forces to avoid slip.

3. Song et al., “Slip Analysis for a Wheeled Mobile Manipulator,” ASME JDSMC, 2018.
   DOI: [10.1115/1.4037287](https://doi.org/10.1115/1.4037287)
   Relevant claim: manipulator motion, reaction wrench, speed and acceleration can induce slip;
   friction constraints are validated through simulation and experiment.

4. Chu et al., “Instantaneous Center of Rotation Tracking Control of Four-Wheel Independent
   Steering Vehicles Under Large-Curvature Turning Conditions,” IEEE T-ITS, 2024.
   DOI: [10.1109/TITS.2024.3432885](https://doi.org/10.1109/TITS.2024.3432885)
   Relevant claim: 4WIS large-curvature path tracking through ICR control.

5. Storms and Tilbury, “Dynamic Weight-Shifting for Improved Maneuverability and Rollover
   Prevention in High-Speed Mobile Manipulators,” ASME JDSMC, 2016.
   DOI: [10.1115/1.4033841](https://doi.org/10.1115/1.4033841)
   Relevant claim: manipulator-based dynamic weight shifting and rollover prevention.

6. Ding et al., “Online Dynamic Tip-Over Avoidance for a Wheeled Mobile Manipulator With an
   Improved Tip-Over Moment Stability Criterion,” IEEE Access, 2019.
   DOI: [10.1109/ACCESS.2019.2915115](https://doi.org/10.1109/ACCESS.2019.2915115)
   Relevant claim: physical tip-over avoidance using arm posture and/or base speed.

These works prevent broad novelty claims such as “arm-based slip avoidance,” “arm-based weight
shifting,” “ICR control,” or “tip-over avoidance” without a narrower irreversible contribution.

## 3. Supporting prior work

- Ben Salem et al., aided locomotion with robot arms:
  [10.1109/AIM.2003.1225100](https://doi.org/10.1109/AIM.2003.1225100)
- Abeykoon and Ohnishi, traction improvement by changing CoG:
  [10.1109/INDIN.2005.1560468](https://doi.org/10.1109/INDIN.2005.1560468)
- Huang et al., manipulator-motion stability compensation:
  [10.1163/156855399X00892](https://doi.org/10.1163/156855399X00892)
- Bostelman and Hong, NIST physical stability measurements:
  [10.6028/NIST.TN.1955](https://doi.org/10.6028/NIST.TN.1955)
- Lee and Li, 4WIS4WID kinematics/dynamics/control:
  [10.1049/joe.2014.0241](https://doi.org/10.1049/joe.2014.0241)

## 4. Official venue criteria

- IEEE Access scope and applied/negative-result categories:
  <https://ieeeaccess.ieee.org/about/>
- IEEE Access author guidelines:
  <https://ieeeaccess.ieee.org/authors/submission-guidelines/>
- IEEE Robotics and Automation Letters scope:
  <https://www.ieee-ras.org/publications/ieee-robotics-and-automation-letters/>
- RA-L author information and page limits:
  <https://www.ieee-ras.org/publications/ra-l/ra-l-information-for-authors/>
- IEEE Transactions on Robotics author criteria:
  <https://www.ieee-ras.org/publications/t-ro/t-ro-information-for-authors/>

## 5. Exclusions and search limits

- arXiv-only and MDPI papers are not used as the evidentiary basis.
- Subscription article PDFs are not redistributed.
- The literature review identifies direct novelty collisions but is not claimed to contain every
  Google Scholar result.
- Oracle/Web ChatGPT is advisory and is archived separately from primary evidence.
