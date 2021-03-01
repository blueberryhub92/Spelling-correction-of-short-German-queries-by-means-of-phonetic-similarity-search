# Spelling-correction-of-short-German-queries-by-means-of-phonetic-similarity-search

This repository includes the scripts for evaluating the correction of spelling errors found in user search queries.

Evaluation_Phonetic_Encoded.zip contains the script for the version that uses a phonetically converted dictionary and phonetically converted user search query to perform the correction. The used dictionaries is contained in the file.

Evaluation_Not_Encoded.zip contains the script for the version that uses only the original user search query and the original dictionary entries. The used dictionary is contained in the file.

Error_List.csv contains the list of 967 wrongly spelled words, as well as their correct forms and the errors encountered in the individual words.

User_Search_Queries.csv contains the list of 264793 search queries.

Google_Docs_Error_List.txt contains the error list including the suggestions made by Google Docs.

Hunspell_Error_List.txt contains the error list including the suggestions made by Hunspell.
