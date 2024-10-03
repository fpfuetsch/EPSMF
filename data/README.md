# Data

## Structure of this directory

- one directory for each exercise which contains again one directory for each exercise attempt as well as the sample solution
- `edit.log.anon` (log of attempt entries -- saving the code produces an entry)
- `exam.csv.anon` (final exam results for each student)
- `prior_xp.csv.anon` (prior coding experience in months for each student)
- `testat.csv.anon` (results for the three programming exams held during the term)
- `solution_metadata.csv` (metadata from parsing the sample solutions' abstract syntax tree)

## Structure of edit.log.anon (provided by IDE Plugin)

- 0 name of exercise
- 1 identifier of attempt (matches with subdirectories)
- 2 exercise difficulty
- 3 --unused--
- 4 code length (number of characters)
- 5 McCabe complexity
- 6 number of passed unit tests
- 7 duration in minutes
- 8 --unused--
- 9 --unused--
- 10 --unused--
- 11 --unused--
- 12 --unused--
- 13 start of attempt (UNIX timestamp)
- 14 --unused--
- 15 failed unit-tests
- 16 timestamp of last save (UNIX timestamp)
- 17 timestamp of last progress regarding unit tests (UNIX timestamp)
- 18 student identifier
