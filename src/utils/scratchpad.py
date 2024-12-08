# # # Usage
# olmo = OLMoHandler()
# response = olmo.generate("""
#                          Generate a text explaining the relationship between different thyroid hormones.
#                          Write the text as short sentences and so, that the logical relationship between two sentences following each other can be easily classified. 
#                          Possible relationships between sentences are: [Causal, Conditional, Sequential, Comparison, Contradiction, Explanation, Definition]. 
#                          Mark the relationships between the sentences using '-[kind of relationship]-'
#                          """)

# response1 = olmo.generate("""
#                             Generate a text about the thyroid and the different thyroid hormones.
#                             Write the text as short sentences and so, that there is a logical relationship (Causal, Conditional, Sequential, Comparison, Contradiction, Explanation, Definition) between two sentences following each other.
#                          """)
# print(response1)
# # prompt2 = f"The text is: '{response1}'." + """
# #                             Mark the relationships between the sentences in the text using '-[kind of relationship]-'
# #                             Possible [kind of relationship]: [Causal, Conditional, Sequential, Comparison, Contradiction, Explanation, Definition].
# #                          """
# # response2 = olmo.generate(prompt2)
# # print(response2)
# prompt3 = f"The text is: '{response1}'." + """
#                                             Identify the most important keywords in this text and give a csv file as output in which keywords and their relationships with the other keywords are given.
#                                             In each row give a keyword and its relationship with the other keywords: following the format 'Keyword1, Keyword2, Relationship'.
#                                             Generate three relationships for each keyword.
#                                             Possible kind of relationship: [Causal, Conditional, Sequential, Comparison, Contradiction, Explanation, Definition].
#                                             """
# response3 = olmo.generate(prompt3)
# print(response3)