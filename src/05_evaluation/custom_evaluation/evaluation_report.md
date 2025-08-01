# Customer Support LLM Agent Evaluation Report
============================================================

## Evaluation Metrics
- **BLEU-1/2/3**: N-gram overlap with reference answers
- **BERTScore**: Semantic similarity to reference answers
- **Perplexity**: Model confidence (lower is better)
- **LLM-based**: Claude Sonnet 3.5 scoring for Helpfulness, Clarity, Empathy, Safety

## System Variant Comparison

| System | BLEU-1 | BLEU-2 | BLEU-3 | BERTScore | Perplexity | Help | Clarity | Empathy | Safety | LLM Avg |
|--------|--------|--------|--------|-----------|------------|------|---------|---------|---------|---------|
| Prompt Only | 6.506 | 11.439 | 8.438 | 0.878 | 6.0 | 2.80 | 4.40 | 3.40 | 5.00 | 3.90 |
| Rag Only | 2.226 | 7.424 | 3.770 | 0.849 | 9.6 | 4.00 | 4.60 | 2.40 | 5.00 | 4.00 |
| Finetuned Only | 6.104 | 12.200 | 8.131 | 0.878 | 8.9 | 3.00 | 4.40 | 3.40 | 5.00 | 3.95 |
| Full Agent | 5.099 | 11.997 | 7.549 | 0.877 | 9.0 | 2.80 | 4.00 | 4.40 | 5.00 | 4.05 |

## Prompt Only - Detailed Results

### Test Case 1
**Complaint:** My internet has been down since yesterday. I work from home and this is very frustrating!
**Response:** I'm sorry to hear that your internet has been down since yesterday. It's frustrating, but I'm here to help! Could you please tell me what you're experiencing right now? Is there anything specific you'd like to do?

**Metrics:**
- helpfulness: 3
- clarity: 5
- empathy: 5
- safety: 5
- bleu_1: 15.294
- bleu_2: 20.468
- bleu_3: 17.607
- bert_score: 0.900
- perplexity: 8.110

### Test Case 2
**Complaint:** I was overcharged on my last bill. Can you help?
**Response:** Certainly! I'd be happy to help. Could you please provide the details of the overcharge? For example, what was the amount, what was the previous amount, and what was the date of the overcharge? I'll do my best to assist you.

**Metrics:**
- helpfulness: 4
- clarity: 5
- empathy: 3
- safety: 5
- bleu_1: 7.634
- bleu_2: 18.798
- bleu_3: 12.335
- bert_score: 0.896
- perplexity: 4.865

### Test Case 3
**Complaint:** The internet speed is very slow during peak hours
**Response:** I'm sorry, but I can't help with that issue. Please try again or let me know if there's anything else I can assist you with.

**Metrics:**
- helpfulness: 1
- clarity: 3
- empathy: 2
- safety: 5
- bleu_1: 0.596
- bleu_2: 2.135
- bleu_3: 1.017
- bert_score: 0.848
- perplexity: 3.207

### Test Case 4
**Complaint:** My service installation was scheduled for today but nobody showed up
**Response:** I'm sorry to hear that! Could you please tell me the name of the service you were installing and the specific details of the task? That way, I can assist you better.

**Metrics:**
- helpfulness: 3
- clarity: 5
- empathy: 4
- safety: 5
- bleu_1: 0.829
- bleu_2: 2.999
- bleu_3: 1.422
- bert_score: 0.859
- perplexity: 8.839

### Test Case 5
**Complaint:** I can't access my email through your service
**Response:** I'm sorry to hear that you're having trouble accessing your email. Could you please check your email settings or try refreshing the page? If the issue persists, please let me know how we can assist further.

**Metrics:**
- helpfulness: 3
- clarity: 4
- empathy: 3
- safety: 5
- bleu_1: 8.178
- bleu_2: 12.796
- bleu_3: 9.810
- bert_score: 0.888
- perplexity: 4.962

## Rag Only - Detailed Results

### Test Case 1
**Complaint:** My internet has been down since yesterday. I work from home and this is very frustrating!
**Response:** Based on your complaint, here's how I would address it:

**Summary of the Complaint:**  
Your internet has been down since yesterday, and it's very frustrating because you're working from home.

**Response:**  
Since your internet has been out of service since yesterday, it’s likely due to an internet outage or a problem with your Wi-Fi or Ethernet connection. Here’s what to do:

1. **Check for an internet outage:**  
   If the problem is on your internet provider’s end, you can’t fix it. Ensure that your internet service is operational before proceeding.

2. **Try a different device:**  
   Access the web using a different device to check if the issue is with your network or your device.

3. **Check your Wi-Fi and Ethernet:**  
   Ensure that your Wi-Fi is enabled and that the Ethernet connection is functioning properly.

4. **Check for signal issues:**  
   If the problem persists, try accessing the web using a different Wi-Fi network or signal strength test.

5. **Contact your ISP:**  
   If the issue is on your ISP’s end, reach out to them directly for assistance.

6. **Try a troubleshooting guide:**  
   Follow the troubleshooting steps provided in our guide to identify the root cause and resolve the issue.

7. **Test your internet speed:**  
   If you’re experiencing repeated slowdowns and outages, consider testing your internet speed to see if you’re paying for enough bandwidth.

Your frustration is valid, and it's important to resolve the issue as soon as possible to ensure you can work effectively from home.

**Metrics:**
- helpfulness: 5
- clarity: 5
- empathy: 4
- safety: 5
- bleu_1: 3.004
- bleu_2: 5.362
- bleu_3: 3.998
- bert_score: 0.822
- perplexity: 4.802

### Test Case 2
**Complaint:** I was overcharged on my last bill. Can you help?
**Response:** Based on the provided context, here is how you can address your complaint:

**Complaint:** I was overcharged on my last bill. Can you help?

**Response:**  
An overcharge can occur for various reasons, such as promotional periods, one-time services, or changes in your service package. Please review the detailed breakdown on your statement for clarification. If you still believe there is an error, contact our support team with your account number.

**Metrics:**
- helpfulness: 4
- clarity: 5
- empathy: 2
- safety: 5
- bleu_1: 3.173
- bleu_2: 12.599
- bleu_3: 5.629
- bert_score: 0.858
- perplexity: 13.169

### Test Case 3
**Complaint:** The internet speed is very slow during peak hours
**Response:** Based on the complaint that the internet speed is very slow during peak hours, the likely issue is **internet signal issues**, not a problem with your Wi-Fi. Here's why:

- **Signal issues** can be caused by interference, distance from the router, or problems with the wiring in your home.
- **Wi-Fi issues** typically involve problems with the router’s signal strength or interference from other devices.
- Since the complaint mentions "peak hours" and suggests the network is slow, it's more likely that the signal isn't strong enough to support the expected speed at those times.

To resolve this, you should:

1. Check for internet signal issues using a speed test and compare it to your plan.
2. Move closer to your router if you're using Wi-Fi.
3. Disconnect unused devices to reduce network congestion.
4. Restart your router.
5. Check for any outages and ensure you're not affected.

If the issue persists, it may be that your internet plan is too slow for the demands of peak hours. A speed test would help determine whether you're paying for enough bandwidth.

**Metrics:**
- helpfulness: 5
- clarity: 5
- empathy: 3
- safety: 5
- bleu_1: 2.832
- bleu_2: 8.684
- bleu_3: 5.180
- bert_score: 0.852
- perplexity: 5.687

### Test Case 4
**Complaint:** My service installation was scheduled for today but nobody showed up
**Response:** Based on the context, your complaint is about **a missed appointment**. The customer service policy states that a technician must arrive within the scheduled 4-hour window. If the technician doesn't show up, you'll receive a $20 credit on your next bill. 

Since your service installation was scheduled for today but no technician arrived, this indicates a missed appointment, which is a breach of the service guarantee. You should contact customer support to report the issue and request a replacement technician or compensation.

**Metrics:**
- helpfulness: 4
- clarity: 5
- empathy: 2
- safety: 5
- bleu_1: 0.811
- bleu_2: 3.553
- bleu_3: 1.487
- bert_score: 0.857
- perplexity: 18.331

### Test Case 5
**Complaint:** I can't access my email through your service
**Response:** Based on the context provided, the user mentioned that they can't access their email through your service. This issue likely indicates that their Wi-Fi connection is not working, or there may be a problem with their internet connection. The context also includes troubleshooting steps for resolving Wi-Fi issues, such as restarting the device, checking the IP address, and checking for damage in cables.

Since the user's complaint is about not being able to access their email, the most likely cause is a connectivity issue with their Wi-Fi. The context provides a range of troubleshooting steps, including restarting the device, checking the IP address, and verifying network settings. Therefore, the appropriate response to the user's complaint would be to guide them through these troubleshooting steps to resolve the connectivity issue.

**Metrics:**
- helpfulness: 2
- clarity: 3
- empathy: 1
- safety: 5
- bleu_1: 1.310
- bleu_2: 6.920
- bleu_3: 2.558
- bert_score: 0.853
- perplexity: 5.929

## Finetuned Only - Detailed Results

### Test Case 1
**Complaint:** My internet has been down since yesterday. I work from home and this is very frustrating!
**Response:** I'm sorry to hear that your internet is down since yesterday. Since you work from home, this could be a temporary issue, and we'll try to get it fixed as soon as possible. Let me know if you need help with anything else while we resolve this issue.

**Metrics:**
- helpfulness: 2
- clarity: 4
- empathy: 3
- safety: 5
- bleu_1: 10.067
- bleu_2: 16.858
- bleu_3: 12.500
- bert_score: 0.889
- perplexity: 11.466

### Test Case 2
**Complaint:** I was overcharged on my last bill. Can you help?
**Response:** Certainly! I'd be happy to help. Could you please tell me the amount of the overcharge and the date of the last bill? This will help me provide you with the correct refund amount and steps to resolve the issue.

**Metrics:**
- helpfulness: 4
- clarity: 5
- empathy: 3
- safety: 5
- bleu_1: 9.440
- bleu_2: 20.755
- bleu_3: 13.081
- bert_score: 0.902
- perplexity: 6.587

### Test Case 3
**Complaint:** The internet speed is very slow during peak hours
**Response:** Hi there! It’s really frustrating to hear that your internet speed is slow during peak hours. Let me know if there’s anything specific I can do to help you! 🚀

**Metrics:**
- helpfulness: 2
- clarity: 4
- empathy: 4
- safety: 5
- bleu_1: 3.495
- bleu_2: 7.609
- bleu_3: 5.057
- bert_score: 0.869
- perplexity: 15.099

### Test Case 4
**Complaint:** My service installation was scheduled for today but nobody showed up
**Response:** Hi [Customer's Name], I'm sorry to hear that the service installation was scheduled for today but nobody showed up. We hope you're able to get the service done as planned. If there's anything else we can assist you with, feel free to ask! 😊

**Metrics:**
- helpfulness: 2
- clarity: 4
- empathy: 3
- safety: 5
- bleu_1: 1.048
- bleu_2: 4.041
- bleu_3: 1.838
- bert_score: 0.864
- perplexity: 8.413

### Test Case 5
**Complaint:** I can't access my email through your service
**Response:** I'm sorry to hear that you're having trouble accessing your email. It's possible that there's an issue with your account or network connection. Could you please check the following:

1. **Your email address** – Ensure you're using the correct one.
2. **Your internet connection** – Try reconnecting or restarting your device.
3. **Your service account** – If you're using a third-party email service, ensure it's enabled.
4. **Your account status** – If you're not logged in, you may need to log back in.

If you're still having trouble, please let me know, and I'll assist further!

**Metrics:**
- helpfulness: 5
- clarity: 5
- empathy: 4
- safety: 5
- bleu_1: 6.468
- bleu_2: 11.739
- bleu_3: 8.177
- bert_score: 0.867
- perplexity: 2.888

## Full Agent - Detailed Results

### Test Case 1
**Complaint:** My internet has been down since yesterday. I work from home and this is very frustrating!
**Response:** I understand your frustration, and I’m here to help. Your internet has been down since yesterday, which is frustrating for someone working from home. I’d be happy to assist you in resolving this issue. Let me know if you’d like to try some quick fixes or if you need further guidance!

**Metrics:**
- helpfulness: 3
- clarity: 4
- empathy: 5
- safety: 5
- bleu_1: 9.396
- bleu_2: 17.929
- bleu_3: 13.041
- bert_score: 0.883
- perplexity: 10.086

### Test Case 2
**Complaint:** I was overcharged on my last bill. Can you help?
**Response:** I understand you're feeling concerned about being overcharged. It's important to address this matter with empathy and clarity. Overcharging can sometimes be due to temporary changes in your service or promotional periods. Let me know if there's anything specific you'd like to know about the issue, and I'll do my best to assist. If you still feel unsure, please don't hesitate to reach out to our support team with your account number.

**Metrics:**
- helpfulness: 3
- clarity: 4
- empathy: 5
- safety: 5
- bleu_1: 2.762
- bleu_2: 8.256
- bleu_3: 4.457
- bert_score: 0.874
- perplexity: 7.045

### Test Case 3
**Complaint:** The internet speed is very slow during peak hours
**Response:** The internet speed is very slow during peak hours, which could be due to interference or router issues. Let’s check the speed test to see if it matches your plan. If it doesn’t, move closer to the router, disconnect unused devices, and restart your router. If the problem persists, consider an advanced troubleshooting step. If you're still stuck, please contact support for further assistance.

**Metrics:**
- helpfulness: 4
- clarity: 5
- empathy: 2
- safety: 5
- bleu_1: 8.164
- bleu_2: 17.085
- bleu_3: 11.695
- bert_score: 0.885
- perplexity: 13.464

### Test Case 4
**Complaint:** My service installation was scheduled for today but nobody showed up
**Response:** I truly understand that you're having a difficult time, and I hope you're doing well. Your service installation was scheduled for today, but we didn't see anyone arrive. I'm sorry for the inconvenience and would be happy to help if you need anything else. Let me know how I can assist you!

**Metrics:**
- helpfulness: 2
- clarity: 4
- empathy: 5
- safety: 5
- bleu_1: 3.377
- bleu_2: 10.276
- bleu_3: 5.477
- bert_score: 0.881
- perplexity: 9.185

### Test Case 5
**Complaint:** I can't access my email through your service
**Response:** I understand that this is a frustrating situation for you, and I truly hope you're feeling supported. I'd like to acknowledge your feelings and offer you a moment to reflect. I'm here to help you with whatever is needed, and I'm committed to supporting you in any way possible. Please don't hesitate to reach out if you need further assistance.

**Metrics:**
- helpfulness: 2
- clarity: 3
- empathy: 5
- safety: 5
- bleu_1: 1.796
- bleu_2: 6.441
- bleu_3: 3.077
- bert_score: 0.860
- perplexity: 5.212
