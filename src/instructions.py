"""Library of instructions."""
import collections
import json
import random
import re
import string
from typing import Dict, Optional, Sequence, Union
import nltk
import spacy
from spacy.cli import download
import emoji
import syllapy
import unicodedata
from collections import Counter
import csv
import io
from absl import logging
import langdetect

from src import instructions_util


_InstructionArgsDtype = Optional[Dict[str, Union[int, str, Sequence[str]]]]

_LANGUAGES = instructions_util.LANGUAGE_CODES

# The relational operation for comparison.
_COMPARISON_RELATION = ("at most", "at least", "around", "less than")

# The maximum number of sentences.
_MAX_NUM_SENTENCES = 20

# The number of placeholders.
_NUM_PLACEHOLDERS = 4

# The number of bullet lists.
_NUM_BULLETS = 5

# The options of constrained response.
_CONSTRAINED_RESPONSE_OPTIONS = (
"My answer is yes.", "My answer is no.", "My answer is maybe.")

# The options of starter keywords.
_STARTER_OPTIONS = ("I would say", "My answer is", "I believe",
"In my opinion", "I think", "I reckon", "I feel",
"From my perspective", "As I see it", "According to me",
"As far as I'm concerned", "To my understanding",
"In my view", "My take on it is", "As per my perception")

# The options of ending keywords.
# TODO(jeffreyzhou) add more ending options
_ENDING_OPTIONS = ("Any other questions?",
"Is there anything else I can help with?")

# The number of highlighted sections.
_NUM_HIGHLIGHTED_SECTIONS = 4

# The section spliter.
_SECTION_SPLITER = ("Section", "SECTION", "Part", "PART", "Paragraph", "PARAGRAPH", "CHAPTER")

# The number of sections.
_NUM_SECTIONS = 5

# The number of paragraphs.
_NUM_PARAGRAPHS = 5

# The postscript marker.
_POSTSCRIPT_MARKER = ("P.S.", "P.P.S")

# The number of keywords.
_NUM_KEYWORDS = 2

# The occurrences of a single keyword.
_KEYWORD_FREQUENCY = 3

# The occurrences of a single letter.
_LETTER_FREQUENCY = 10

# The occurrences of words with all capital letters.
_ALL_CAPITAL_WORD_FREQUENCY = 20

# The number of words in the response.
_NUM_WORDS_LOWER_LIMIT = 100
_NUM_WORDS_UPPER_LIMIT = 500


class Instruction:
	"""An instruction template."""

	def __init__(self, instruction_id):
		self.id = instruction_id
  
	def check_following(self, value, **kwargs):
		raise NotImplementedError("`check_following` not implemented.")


class ResponseLanguageChecker(Instruction):
	def check_following(self, value, *, language=None):
		"""CCheck if the primary language of the response follows the instruction.
		
  		Description pattern:
			Your response should be primarily in {language} language.

		Note:
            This uses `langdetect` to identify the **primary language** of the text.
            It does not guarantee that every single word is in the specified language,
            only that the main language of the text matches `language`. In this way, although
			the prompt may request a response in a specific language, the response can be 
			in a different language as long as the main language is the same.
            
		Args:
			value: A string representing the response.
			language: Expected language for the response. 

		Returns:
			True if the language of `value` follows instruction; otherwise False.
		"""
		self._language = language
	
		try:
			import langdetect
			return langdetect.detect(value) == self._language
		except langdetect.LangDetectException as e:
			import logging
			logging.error("Unable to detect language for text %s due to %s", value, e)
		return False

class NumberOfSentences(Instruction):
	def check_following(self, value,  *, num_sentences = None, relation = None):
		"""Check if the number of sentences follows the instruction.

		Description pattern:
			Your response should contain {relation} {num_sentences} sentences.
  
		Args:
			value: A string representing the response.
			num_sentences: An integer specifying the number of sentences as a
				threshold.
			relation: A string in (`at most`, `at least`, `around`, `less than`), defining the relational
				operator for comparison.
			Four relational comparisons are supported for now:
			if 'at most', the actual number of sentences <= the threshold;
			if 'at least', the actual number of sentences >= the threshold.
			if 'around', the actual number of sentences is within 2 of the threshold.
			if 'less than', the actual number of sentences < the threshold.

		Returns:
			True if the response follows the instruction.

		Raise:
			ValueError if the string in `instruction_args` is not in
				[`less than`, `at least`, `at most`, `around`].
		"""
  
		self._num_sentences_threshold = num_sentences
		self._comparison_relation = relation
   
		num_sentences = instructions_util.count_sentences(value)
		if self._comparison_relation == "at most":
			return num_sentences <= self._num_sentences_threshold
		elif self._comparison_relation == "at least":
			return num_sentences >= self._num_sentences_threshold
		elif self._comparison_relation == "around":
			return abs(num_sentences - self._num_sentences_threshold) <= 2
		elif self._comparison_relation == "less than":
			return num_sentences < self._num_sentences_threshold



class PlaceholderChecker(Instruction):
	def check_following(self, value, *, num_placeholders = None):
		"""Check if the number of placeholders follows the instruction.
		
  		Description pattern:
			The response must contain at least {num_placeholders} placeholders represented by square brackets, such as [address].
			Placeholders are represented with square brackets, e.g., [this is a placeholder].
		
  		Args:
			value: A string representing the response.
			num_placeholders: An integer denoting the minimum number of
				placeholders required in the response.
		Returns:
			True if the actual number of placeholders in the response is greater than
				or equal to `num_placeholders`; otherwise, False.
		"""
		self._num_placeholders = num_placeholders
  
		import re
		
		placeholders = re.findall(r"\[.*?\]", value)
		num_placeholders = len(placeholders)
		return num_placeholders >= self._num_placeholders


class BulletListChecker(Instruction):
	def check_following(self, value, *, num_bullets = None):
		"""Check if the number of bullet lists meets the requirement.
		
  		Description pattern:
			Your answer must contain exactly {num_bullets} bullet points.
			Use the markdown bullet points such as:
			* This is point 1.
			* This is point 2

		Args:
			value: A string representing the response. The response is expected to
				contain some bullet lists that start with `\*`.
			num_bullets: An integer specifying the exact number of bullet lists
				that is required to appear in the response.
		Returns:
			True if the actual number of bullet lists in the response meets the
				requirement.
		"""
		self._num_bullets = num_bullets

		import re
		
		bullet_lists = re.findall(r"^\s*\*[^\*].*$", value, flags=re.MULTILINE)
		bullet_lists_2 = re.findall(r"^\s*-.*$", value, flags=re.MULTILINE)
		num_bullet_lists = len(bullet_lists) + len(bullet_lists_2)
		return num_bullet_lists == self._num_bullets


class ConstrainedResponseChecker(Instruction):
	def check_following(self, value):
		"""Checks if the response matches the constrained options.
		
		Description pattern:
			Answer with one of the following options: {response_options}
   
		Args:
			value: A string representing the response.

		Returns:
			True if the actual response contains one of the options in the constrained responses; otherwise False.
		"""
		self._constrained_responses = (
			"My answer is yes.", "My answer is no.", "My answer is maybe.")

		value = value.strip()
		for constrained_response in self._constrained_responses:
			if constrained_response in value:
				return True
		return False


class ConstrainedStartChecker(Instruction):
	def check_following(self, value,  *, starter = None):
		"""Checks if the response starts with the constrained keyword or phrase.
  
		Description pattern:
			During the conversation, when it is your turn, please always start with {starter}.

		Args:
			starter: A string representing the keyward that the response should start with.
			value: A string representing the response.

		Returns:
			True if the response starts with the given phrase or keyword that is
				contained in `instruction_args`; otherwise, False.
		"""
		self._starter = starter.strip() if isinstance(starter, str) else starter


		response_pattern = r"^\s*" + self._starter + r".*$"
		response_with_constrained_start = re.search(response_pattern, value, flags=re.MULTILINE)
		return True if response_with_constrained_start else False


class HighlightSectionChecker(Instruction):
	def check_following(self, value, *, num_highlights=None):
		"""Checks if the number of highlighted sections meets the requirement.
  
		Description pattern:
			Highlight at least {num_highlights} sections in your answer with markdown, i.e. *highlighted section*.
   
		Args:
			value: a string repesenting the response. The response is expected to
				contain highlighted sections in the format of *highlighted*.
			num_highlights: Minimum number of highlighted sections required.


		Returns:
			True if the actual number of highlighted sections in the format of
			*highlighed sections* meets the minimum requirement; otherwise False.
		"""
		self._num_highlights = num_highlights

		num_highlights = 0
		highlights = re.findall(r"\*[^\n\*]*\*", value)
		double_highlights = re.findall(r"\*\*[^\n\*]*\*\*", value)
		for highlight in highlights:
			if highlight.strip("*").strip():
				num_highlights += 1
		for highlight in double_highlights:
			if highlight.removeprefix("**").removesuffix("**").strip():
				num_highlights += 1

		return num_highlights >= self._num_highlights


class SectionChecker(Instruction):
	def check_following(self, value, *, section_spliter = None, num_sections = None):
		"""Checks the response contains multiple sections.

		Args:
			value: A string representing the response. The response is expected
				to contain multiple sections (number of sections is greater than 1).
				A new section starts with `Section 1`, where the number denotes the
				section index.
			section_spliter: A string represents the section spliter keyword that
				marks a new section, i.e., `Section` or `SECTION`.
			num_sections: An integer specifying the minimum number of sections.
		
  		discription pattern:
			Your response must have {num_sections} sections. Mark the beginning of each section with {section_spliter} X, such as:
				{section_spliter} 1
				[content of section 1]
				{section_spliter} 2
				[content of section 2]

		Returns:
			True if the number of sections in the response is greater than or equal to
				the minimum number of sections; otherwise, False.
		"""
		self._section_spliter = section_spliter.strip() if isinstance(section_spliter, str) else section_spliter
		self._num_sections = num_sections
		
		section_splitter_patten = r"\s?" + re.escape(self._section_spliter) + r"\s?\S+\s?"
		sections = re.split(section_splitter_patten, value)

		num_sections = len(sections) - 1
		return num_sections >= self._num_sections


class ParagraphChecker(Instruction):
	def check_following(self, value, *, num_paragraphs = None):
		"""Checks the response contains required number of paragraphs.

		Description pattern:
			There should be {num_paragraphs} paragraphs.
			Paragraphs are separated with the markdown divider: ***.
   
		Args:
			value: A string representing the response. The response may contain
				paragraphs that are separated by the markdown divider: `***`.
			num_paragraphs: An integer specifying the number of paragraphs.
		Returns:
			True if the actual number of paragraphs is the same as required;
				otherwise, False.
		"""
		self._num_paragraphs = num_paragraphs

		paragraphs = re.split(r"\s*\*\*\*\s*", value)
		num_paragraphs = len(paragraphs)

		for index, paragraph in enumerate(paragraphs):
			if not paragraph.strip():
				if index == 0 or index == len(paragraphs) - 1:
					num_paragraphs -= 1
				else:
					return False

		return num_paragraphs == self._num_paragraphs


class PostscriptChecker(Instruction):
	def check_following(self, value, *, postscript_marker = None):
		"""Checks if the response follows the postscript format.

		Description pattern:
			At the end of your response, please explicitly add a postscript starting with {postscript}.
   
		Args:
			value: a string representing the response. The response is expected to
				contain a postscript section.
			postscript_marker: A string containing the keyword that marks the start
				of the postscript section.
    
		Returns:
			True if the response contains a postscript section starting with
			the keyword containing in the `instruction_args`; otherwise False.
		"""
		self._postscript_marker = postscript_marker.strip() if isinstance(postscript_marker, str) else postscript_marker

		value = value.lower()
		if self._postscript_marker == "P.P.S":
			postscript_pattern = r"\s*p\.\s?p\.\s?s.*$"
		elif self._postscript_marker == "P.S.":
			postscript_pattern = r"\s*p\.\s?s\..*$"
		else:
			postscript_pattern = r"\s*" + self._postscript_marker.lower() + r".*$"
		postscript = re.findall(postscript_pattern, value, flags=re.MULTILINE)
		return True if postscript else False


class KeywordChecker(Instruction):
	def check_following(self, value, *, keywords = None):
		"""
  		Check if the response contain the expected keywords.
    
		Description pattern:
			Include keywords {keywords} in the response.
		Args:
			value: A string representing the response.
			keywords: A sequence of strings representing the keywords that are
				expected in the response.

		Returns:
			True if the response contains all the keywords in `instruction_args`;
    	"""
		self._keywords = keywords
		self._keywords = sorted(self._keywords)
   
		for keyword in self._keywords:
			pattern = r"\b" + re.escape(keyword) + r"\b"
			if not re.search(pattern, value, flags=re.IGNORECASE):
				return False
		return True

class KeywordFrequencyChecker(Instruction):
	def check_following(self, value, *, keyword = None, frequency = None, relation = None):
		"""
		Checks if the response contains the keyword with required frequency.
		
		Description pattern:
			In your response, the word {keyword} should appear {relation} {frequency} times.

		Args:
			value: A string representing the response.
			keyword: A string representing a keyword that is expected in the response.
			frequency: An integer specifying the number of times `keyword` is expected
				to appear in the response.
			relation: A string in (`at most`, `at least`, `around`, `less than`), defining the relational
				operator for comparison.
			Four relational comparisons are supported for now:
				if 'at most', the actual number of occurrences <= frequency;
				if 'at least', the actual number of occurrences >= frequency.
				if 'around', the actual number of occurrences is within 5 of frequency.
				if 'less than', the actual number of occurrences < frequency.

		Returns:
			True if the actual number of occurrences of `keyword` in the response
		"""
		self._keyword = keyword.strip()
		self._frequency = frequency
		self._comparison_relation = relation
		
		pattern = r'\b{}\b'.format(re.escape(self._keyword))
		actual_occurrences = len(re.findall(pattern, value, flags=re.IGNORECASE))

		if self._comparison_relation == "at most":
			return actual_occurrences <= self._frequency
		elif self._comparison_relation == "at least":
			return actual_occurrences >= self._frequency 
		elif self._comparison_relation == "around":
			return abs(actual_occurrences - self._frequency) <= 5
		elif self._comparison_relation == "less than":
			return actual_occurrences < self._frequency



class NumberOfWords(Instruction):
	def check_following(self, value, *, num_words = None, relation = None):
		"""
		Checks if the number of words meets the requirement.
		
		Description pattern:
			Your response should contain {relation} {num_words} words.
   
  		Args:
			value: A string representing the response.
			num_words: An integer specifying the number of words contained in the
				response.
			relation: A string in (`at most`, `at least`, `around`, `less than`), defining the relational
				operator for comparison.
				Four relational comparisons are supported for now:
				if 'at most', the actual number of words <= num_words;
				if 'at least', the actual number of words >= num_words.
				if 'around', the actual number of words is within 25 of num_words.
				if 'less than', the actual number of words < num_words.
		Returns:
  			True if the actual number of words in the response meets the requirement;
		"""
		self._num_words = num_words
		self._comparison_relation = relation
   
		num_words = instructions_util.count_words(value)


		if self._comparison_relation == "at most":
			return num_words <= self._num_words
		elif self._comparison_relation == "at least":
			return num_words >= self._num_words  # pytype: disable=bad-return-type
		elif self._comparison_relation == "around":
			return abs(num_words - self._num_words) <= 25
		elif self._comparison_relation == "less than":
			return num_words < self._num_words


class JsonFormat(Instruction):
	def check_following(self, value):
		"""
     	Check the Json format.
	
		Description pattern:
			Your response should be in a valid JSON format.
		Args:
  			value: A string representing the response.	
     
		Returns:
			True if the response is in a valid JSON format; otherwise False.
      	"""
		value = (
			value.strip()
			.removeprefix("```json")
			.removeprefix("```Json")
			.removeprefix("```JSON")
			.removeprefix("```")
			.removesuffix("```")
			.strip()
			)
		try:
			json.loads(value)
		except ValueError as _:
			return False
		return True


class ParagraphFirstWordCheck(Instruction):
	def check_following(self, value, num_paragraphs = None, nth_paragraph = None, first_word = None):
		"""Checks for required number of paragraphs and correct first word.

		Args:
			value: a string representing the response. The response may contain
				paragraphs that are separated by two new lines and the first word of
				the nth paragraph will have to match a specified word.
			
   			num_paragraphs: An integer indicating the number of paragraphs expected
				in the response. A paragraph is a subset of the string that is
				expected to be separated by '\n\n'.
			nth_paragraph: An integer indicating the paragraph number that we look at.
				Note that n starts from 1.
			first_word: A string that represent the first word of the bth paragraph.

		Returns:
			True if the number of paragraphs is the same as required and the first
				word of the specified paragraph is the same as required. Otherwise, false.
		"""
		self._num_paragraphs = num_paragraphs
		self._nth_paragraph = nth_paragraph
		self._first_word = first_word
	
		paragraphs = re.split(r"\n\n", value)
		num_paragraphs = len(paragraphs)

		for paragraph in paragraphs:
			if not paragraph.strip():
				num_paragraphs -= 1

		if self._nth_paragraph <= num_paragraphs:
			paragraph = paragraphs[self._nth_paragraph - 1].strip()
			if not paragraph:
				return False
		else:
			return False

		first_word = ""
		punctuation = {".", ",", "?", "!", "'", '"'}

		word = paragraph.split()[0].strip()
		word = word.lstrip("'")
		word = word.lstrip('"')

		for letter in word:
			if letter in punctuation:
				break
			first_word += letter.lower()

		return (
			num_paragraphs == self._num_paragraphs
			and first_word == self._first_word
		)

class ForbiddenWords(Instruction):
	def check_following(self, value, forbidden_words = None):
		"""
  		Check if the response does not contain the expected keywords.

		Description pattern:
			Do not include keywords {forbidden_words} in the response.

		Args:
			value: A string representing the response.
			forbidden_words: A sequences of strings respresenting words that are not
				allowed in the response.

		Returns:
			A string representing the instruction description.
		"""
		
		self._forbidden_words = list(set(forbidden_words))
		self._forbidden_words = sorted(self._forbidden_words)
		self._description_pattern = (
			"Do not include keywords {forbidden_words} in the response."
		)
   
		for word in self._forbidden_words:
			if re.search(r"\b" + word + r"\b", value, flags=re.IGNORECASE):
				return False
		return True



class TwoResponsesChecker(Instruction):
	def check_following(self, value):
		"""Checks if the response has two different answers.
		
		Description pattern:
			Give two different responses. The responses should
			be separated by 6 asterisk symbols: ******.

			Typically, the prompt itself has implied that the responses should be different.
			These test cases are valid.
   
		Args:
			value: A string representing the response.

		Returns:
			True if two responses are detected and false otherwise.
		"""
		valid_responses = []
		responses = re.split(r"(?<!\*)\*{6}(?!\*)", value)
		for index, response in enumerate(responses):
			if not response.strip():
				if index != 0 and index != len(responses) - 1:
					return False
			else:
				valid_responses.append(response)
		return (
			len(valid_responses) == 2
			and valid_responses[0].strip() != valid_responses[1].strip()
			)


class RepeatPromptThenAnswer(Instruction):
	def check_following(self, value, *, prompt_to_repeat = None):
		"""
		Checks that Prompt is first repeated then answered.
  
		Description pattern:
			First repeat the request word for word without change,
			then give your answer (1. do not say any words or characters
   			before repeating the request; 2. the request you need to repeat	
			does not include this sentence)

			A stricter prompt constraint is allowed only if it can pass the evaluation function and evaluation arguments.
   
		Args:
			value: A string representing the response.
			prompt_to_repeat: The prompt that is meant to be repeated.
   
		Returns:
  			True if the response starts with the prompt that is meant to be repeated.
		"""
		def normalize_text(s: str) -> str:
			s = s.strip().lower()
			return re.sub(rf"[{re.escape(string.punctuation)}]", "", s)

		self._prompt_to_repeat = prompt_to_repeat
		
		norm_value = normalize_text(value)
		norm_prompt = normalize_text(self._prompt_to_repeat)
		if norm_value.startswith(norm_prompt):
			return True
		return False


class EndChecker(Instruction):
	def check_following(self, value, *, end_phrase = None):
		"""
		Checks if the response ends with the expected phrase.
		
		Description pattern:
			Finish your response with this exact phrase {ender}.
			No other words should follow this phrase.	
	
		Args:
			value: A string representing the response.
			end_phrase: A string representing the phrase the response should end with.

		Returns:
			True if the response ends with the given phrase; otherwise, False.
		"""
		self._end_phrase = (
			end_phrase.strip() if isinstance(end_phrase, str) else end_phrase
		)
		
		value = value.strip().strip("\"").lower()
		self._end_phrase = self._end_phrase.strip().lower()
		return value.endswith(self._end_phrase)


class TitleChecker(Instruction):
	def check_following(self, value):
		"""
  		Checks whether the response contains at least one valid title enclosed in double angle brackets.
		
  		Description pattern:
			Your answer must include a title, wrapped in double angular brackets,
			such as <<poem of joy>>.
   
		Args:
			value: A string representing the response.
   
		Returns:
  			True if the response contains at least one non-empty title wrapped in << >>;
			False otherwise.
    	"""
		pattern = r"<<([^<\n][^>\n]*)>>"
		re_pattern = re.compile(pattern)
		titles = re.findall(re_pattern, value)

		for title in titles:
			if title.strip():
				return True
		return False


class LetterFrequencyChecker(Instruction):
	def check_following(self, value, *, letter = None, let_frequency = None, let_relation = None):
		"""
  		Checks that the response contains the letter at the right frequency.
	
		Description pattern:
			In your response, the letter {letter} should appear {let_relation}
			{let_frequency} times.
    
		Args:
			value: A string representing the response.
			letter: A string representing a letter that is expected in the response.
			let_frequency: An integer specifying the number of times `keyword` is
				expected to appear in the response.
			let_relation: A string in (`at most`, `at least`, `around`, `less than`), defining the
				relational operator for comparison. 
    			Four relational comparisons are supported for now; 
       			if 'at most', the actual number of occurrences <= frequency; 
          		if 'at least', the actual number of occurrences >= frequency;
            	if 'around', the actual number of occurrences is within 5 of frequency.
				if 'less than', the actual number of occurrences < frequency.
    	
		Returns:
			True if the actual number of occurrences of `letter` in the response.
     	"""
		
		self._letter = letter.strip().lower()
		self._frequency = let_frequency
		self._comparison_relation = let_relation
   
		value = value.lower()
		letters = collections.Counter(value)

		if self._comparison_relation == "at most":
			return letters[self._letter] <= self._frequency
		elif self._comparison_relation == "at least":
			return letters[self._letter] >= self._frequency
		elif self._comparison_relation == "around":
			return abs(letters[self._letter] - self._frequency) <= 5
		elif self._comparison_relation == "less than":
			return letters[self._letter] < self._frequency



class CapitalLettersEnglishChecker(Instruction):
	def check_following(self, value):
		"""
  		Checks that the response is in English and in all capital letters.
		
  		Description pattern:
			Your entire response should be in English, and in all capital letters.
	
		Args:
			value: A string representing the response.
		
  		Returns:
  			True if the response is in English and in all capital letters; otherwise False.	
    	"""
		try:
			return value.isupper() and langdetect.detect(value) == "en"
		except langdetect.LangDetectException as e:
			logging.error(
			"Unable to detect language for text %s due to %s", value, e
			) 
		return True


class LowercaseLettersEnglishChecker(Instruction):
	def check_following(self, value):
		"""Checks that the response is in English and in all lowercase letters.
	
  		Description pattern:
			Your entire response should be in English, and in all lowercase letters.
			No capital letters are allowed.
		
		Note: Symbols, numbers, and punctuation do not affect the lowercase check.
  
  		Args:
			value: A string representing the response.
   
		Returns:
			True if the response is in English and in all lowercase letters; otherwise False.
  		"""
		import langdetect
		try:
			return value.islower() and langdetect.detect(value) == "en"
		except langdetect.LangDetectException as e:
			logging.error(
			"Unable to detect language for text %s due to %s", value, e
			) 
		return True


class CommaChecker(Instruction):
	def check_following(self, value):
		"""Checks that the response does not contain commas.
		
  		Description pattern:
			Do not use commas in your response.
   
		Args:
  			value: A string representing the response.	
     
		Returns:
			True if the response does not contain commas; otherwise False.
  		"""
		return not re.search(r"\,", value)


class CapitalWordFrequencyChecker(Instruction):
	def check_following(self, value, *, capital_frequency = None, capital_relation = None):
		"""
  		Checks the frequency of words with all capital letters.
		
  		Description pattern:
			In your response, words with all capital letters should appear
			{relation} {frequency} times.
   
		Args:
			value: A string representing the response.
			capital_frequency: An integer that represents the number of words that
				should be in all capital letters.
			capital_relation: A string that is 'at least' or 'at most' or 'around' or 'less than' that refers to
				the frequency.

		Returns:
			True if the actual number of words with all capital letters meets the
    	"""
		# Hyphenated words will count as one word
		self._frequency = capital_frequency
		self._comparison_relation = capital_relation

		words = instructions_util.nltk.word_tokenize(value)
		capital_words = [word for word in words if word.isupper()]

		capital_words = len(capital_words)

		if self._comparison_relation == "at most":
			return capital_words <= self._frequency
		elif self._comparison_relation == "at least":
			return capital_words >= self._frequency
		elif self._comparison_relation == "around":
			return abs(capital_words - self._frequency) <= 5
		elif self._comparison_relation == "less than":
			return capital_words < self._frequency



class QuotationChecker(Instruction):
	def check_following(self, value):
		"""Checks if the response is wrapped with double quotation marks.
  
		Description pattern:
			Wrap your entire response with double quotation marks.
   
		Args:
			value: A string representing the response.
   
		Returns:
  			True if the response is wrapped with double quotation marks; otherwise False.	
  		"""
		value = value.strip()
		return len(value) > 1 and value[0] == '"' and value[-1] == '"'

