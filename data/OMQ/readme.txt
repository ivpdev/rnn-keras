* Dataset content 

The dataset consists of a set of German emails and online requests from customers to the support center of a multimedia software company. The customer interactions contain questions concerning the company's products. In order to eliminate all information which traces back to the software company and to anonymize personal data, all original requests were modified by 

a) transforming the product domain of the real company to a different product domain of an imaginary company called WAREHOUSE. WAREHOUSE provides management software for online auction sales. This domain transformation required the modification (and partially, the deletion) of all hints to the original product, like product names, software functions, system logs etc. 

b) anonymizing personal data (names or addresses of customers and employees etc.).

One or more problem categories are assigned to each email. These categories represent the general problems which are described in the customer emails. 

* File omq_public_emails.xml

This file contains the email dataset. Each email consists of the email text along with meta-information. In each email text, one or more relevant text parts are marked (i.e., the problem-containing part of the email), together with the category ID of the corresponding category. 

* File omq_public_categories.xml

This file lists all categories that are assigned to the emails in the email dataset. As the categories do not exclude each other in every case, they are combined into category groups of similar categories. Each category consists of an ID and a text description of the problem. 

* Additional information

** Distribution: Public (Distribution for research purposes under a Creative Commons license Attribution-NonCommercial-ShareAlike)

** Dimension/size: 627 emails (containing 638 relevant texts), 41 categories (arranged in 20 groups)

**Time span: The data was collected between 01.01.2011 and 01.01.2012
