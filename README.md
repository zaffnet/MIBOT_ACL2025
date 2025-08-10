
# MIBot - Motivational Interviewing for Smoking Cessation Dataset, based on MIBoT Version 6.3A.

This repository contains the dataset from the study "[A Fully Generative Motivational Interviewing Counsellor Chatbot for Moving Smokers Towards the Decision to Quit](https://arxiv.org/abs/2505.17362)". The dataset comprises annotated transcripts and surveys (including self-reported readiness to quit smoking) from 106 conversations between human smokers and MIBot v6.3A &mdash; a motivational interviewing (MI) chatbot built using OpenAI's GPT-4o.

## Dataset Usage
Researchers and practitioners can use this dataset to: 

    1.  Analyze the language of MI-based counseling sessions,
    2.  Evaluate the relationship between dialogue features and behavior change outcomes, and
    3.  Train or benchmark new conversational agents on MI-style interactions. 
    
We provide both conversation transcripts, broken down by utterance, and pre and post-conversation survey data.

## Citation

If you use this dataset, please cite:

```bibtex
@inproceedings{mahmood-etal-2025-fully,
    title = "A Fully Generative Motivational Interviewing Counsellor Chatbot for Moving Smokers Towards the Decision to Quit",
    author = "Mahmood, Zafarullah  and
      Ali, Soliman  and
      Zhu, Jiading  and
      Abdelwahab, Mohamed  and
      Collins, Michelle Yu  and
      Chen, Sihan  and
      Zhao, Yi Cheng  and
      Wolff, Jodi  and
      Melamed, Osnat C.  and
      Minian, Nadia  and
      Maslej, Marta  and
      Cooper, Carolynne  and
      Ratto, Matt  and
      Selby, Peter  and
      Rose, Jonathan",
    booktitle = "Findings of the Association for Computational Linguistics: ACL 2025",
    month = jul,
    year = "2025",
    address = "Vienna, Austria",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2025.findings-acl.1283/",
    doi = "10.18653/v1/2025.findings-acl.1283",
    pages = "25008--25043",
    ISBN = "979-8-89176-256-5"
}
```

## Overview of the Dataset
The dataset consists of a CSV file (`data.csv`), where each row corresponds to a unique participant. We also provide conversation transcripts between MIBot and participants in another CSV file (`conversations.csv`). 

---

### Description of the Columns in `data.csv`
| **Column Name**                 | **Description**                                                                 |
|--------------------------------|---------------------------------------------------------------------------------|
| **Basic**                      |                                                                                 |
| `ParticipantId`                | Unique Participant ids assigned by researchers.                                              |
| **Pre-conversation Survey on Heaviness of Smoking** |                                                          |
| `DailyNum`                     | How many cigarettes do you typically smoke per day?                            |
| `FirstCig`                     | How soon after you wake up do you smoke your first cigarette?                  |
| `HeavinessOfSmokingIndex`      | Heaviness of Smoking Index                                  |
| **Pre-conversation Survey on Quit Attempts a Week Prior** |                                            |
| `PreConvoQuitAttempt`         | Have you made any quit attempts (meaning consciously not smoking for a specific period of time greater than 24 hours) during the previous week? |
| `PreConvoNumQuitAttempts`     | How many attempts to quit did you make?                                        |
| **Pre-conversation Readiness Rulers** |                                                                            |
| `PreRulerImportance`          | On a scale from 0 to 10, how important is it to you right now to stop smoking? |
| `PreRulerConfidence`          | On a scale from 0 to 10, how confident are you that you would succeed at stopping smoking if you start now? |
| `PreRulerReadiness`           | On a scale from 0 to 10, how ready are you to start making a change at stopping smoking right now? |
| **Post-conversation Readiness Rulers** |                                                                           |
| `PostRulerImportance`         | On a scale from 0 to 10, how important is it to you right now to stop smoking? |
| `PostRulerConfidence`         | On a scale from 0 to 10, how confident are you that you would succeed at stopping smoking if you start now? |
| `PostRulerReadiness`          | On a scale from 0 to 10, how ready are you to start making a change at stopping smoking right now? |
| **Post-conversation Feedback** |                                                                               |
| `FeedbackQ1`                  | What are three words that you would use to describe the chatbot?              |
| `FeedbackQ2`                  | What would you change about the conversation?                                  |
| `FeedbackQ3`                  | Did the conversation help you realize anything about your smoking behavior? Why or why not? |
| `LikedBot`                    | Whether the participant liked `MIBot`, based on responses to `FeedbackQ1-3`.|
| `FoundBotHelpful`             | Whether the participant found `MIBot` helpful, based on responses to `FeedbackQ1-3`. |
| **CARE Survey**               |                                                                                 |
| `CAREQ1` to `CAREQ10`         | Responses to CARE questions. See Section H in the paper for CARE questions. |
| **Week Later Readiness Rulers** |                                                                              |
| `WeekLaterRulerImportance`    | On a scale from 0 to 10, how important is it to you right now to stop smoking? |
| `WeekLaterRulerConfidence`    | On a scale from 0 to 10, how confident are you that you would succeed at stopping smoking if you start now? |
| `WeekLaterRulerReadiness`     | On a scale from 0 to 10, how ready are you to start making a change at stopping smoking right now? |
| **Week Later Quit Attempts**  |                                                                                 |
| `WeekLaterQuitAttempt`        | Have you made any quit attempts (meaning consciously not smoking for a specific period of time greater than 24 hours) during the previous week? |
| `WeekLaterNumQuitAttempts`    | How many attempts to quit did you make?                                        |
| **AutoMISC Labels**           |                                                                                 |
| `AutoMISC_MICO` to `AutoMISC_C:S` | See Section 4.3 in the paper for AutoMISC labels.             |

---

### Description of the Columns in `conversations.csv`

| **Column Name**        | **Description**                                                                                  |
|------------------------|--------------------------------------------------------------------------------------------------|
| `ParticipantID`        | Unique Participant ids we assign.                                                                |
| `Speaker`              | Indicates whether the speaker is the `counsellor` (i.e., `MIBot`) or the `client`.           |
| `Volley#`              | Serial number of the volley in the transcript. "A volley is an uninterrupted utterance or sequence of utterances by one party, before another party speaks." |
| `Utterance#`           | Serial number of the utterance in the transcript.                                                |
| `CumulativeVolley`     | Represents the volley up to utterance # `Utterance#`. The `CumulativeVolley` corresponding to the last utterance of the volley is the complete volley, which can be used to generate the transcript. |
| `Utterance`            | "An utterance is a complete thought, or a thought unit."                                   |
| `AutoMISCLabel`        | Utterance label according to AutoMISC. It can be one of the following: `R`, `Q`, `Other`, `C`, `N`, ... (See Section 4.3 in the paper.) |
| `AutoMISCExplanation`  | Explanation provided by the AutoMISC LLM as part of its chain-of-thought.                        |


## Licensing
This dataset is released under the [Creative Commons Attribution-ShareAlike 4.0 International License (CC BY-SA 4.0)](https://creativecommons.org/licenses/by-sa/4.0/).


## Ethics Approval
This study involved human participants and was conducted with appropriate ethical oversight. The research protocol was approved by the University of Toronto Research Ethics Board (Protocol #49997) on August 3, 2024. All participants provided informed consent prior to taking part in the chatbot conversation. 
All data provided by participants has been de-identified using the [spaCy](https://spacy.io/universe/project/scrubadub_spacy) (version 3.8.4) and [scrubadub](https://github.com/LeapBeyond/scrubadub) (version 2.0.0) Python libraries. Further, the participants self-reported all the columns in the dataset (except for AutoMISC annotations).


## Contact
For inquiries, please contact:

* Jonathan Rose, Professor of Electrical and Computer Engineering, University of Toronto
* Email: [jonathan.rose@utoronto.ca](mailto:jonathan.rose@utoronto.ca)




