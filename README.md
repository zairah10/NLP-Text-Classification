# NLP-Text-Classification

The goal of this project is to classify text data (binary/multi-class) using Machine Learning, Deep Learning and NLP techniques.

The experiments have been conducted on The Consumer Finance Complaints published by the Consumer Financial Protection Bureau.
(https://www.consumerfinance.gov/data-research/consumer-complaints/)
Each week the Consumer Financial Protection Bureau sends thousands of consumers’ complaints about financial products and services to companies for response. Those complaints are published after the company responds or after 15 days, whichever comes first. By adding their voice, consumers help improve the financial marketplace. 

The dataset consists of various columns, but for the purpose of this study only two columns were extracted, namely 'product' and 'consumer_narrative'.

Example

| product | consumer_narrative |
| --- | --- |
| Bank account | I used my Paypal account to buy an item from XXXX. The product did not show up until 6 weeks later and was faulty. I went to Paypal seeking a refund from the Vendor and was denied because it was over thirty days in billing. I 'm out of {$12.00} and most likely so are thousands of other people. |
| Mortgage | I submmited a fully document Request for a Loan Modification with mortgage company M & T Mortgage XXXX Investor XXXX XXXX XXXX to reduce my mortgage payment ; instead to consider my case for a HAMP review, they approved me for a Streamline Loan Mod that reduced my mortgage payment to 43 % of my monthly gross income ( {$2200.00} ). Also they decided to charge me {$980.00} modification fees. |
