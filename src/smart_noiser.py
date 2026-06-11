import sys
import random
import numpy as np

from typing import List, Dict, Any, Tuple
from src.vocab import CASE, ENG, RULE_BASED, ADJACENT, WEIGHTED

sys.setrecursionlimit(1500)

class AzerbaijaniTextNoiser:

    def __init__(
        self,
        eng_prob: float = 0.1,
        rule_based_prob: float = 0.3,
        weighted_prob: float = 0.5,
        case_prob: float = 0.003,
        adjacent_prob: float = 0.5,
        swap_prob: float = 0.001,
        remove_space: float = 0.005
    ) -> None:

        # External vocabularies for noise transformations
        self.ENG: Dict[str, List[str]] = ENG
        self.RULE_BASED: Dict[str, str] = RULE_BASED
        self.WEIGHTED: Dict[str, Dict[str, float]] = WEIGHTED
        self.ADJACENT: Dict[str, List[str]] = ADJACENT
        self.CASE: Dict[str, str] = CASE

        # Counter for recursion
        self.recursion_counter: int = 0
      
        # Noise probabilities
        self.eng_prob: float = eng_prob
        self.rule_based_prob: float = rule_based_prob
        self.weighted_prob: float = weighted_prob
        self.case_prob: float = case_prob
        self.adjacent_prob: float = adjacent_prob
        self.swap_prob = swap_prob
        self.remove_space = remove_space

    def _edit_distance(self, s1: str, s2: str) -> int:
        """
        Compute Levenshtein edit distance between two strings.
        """
        if len(s1) > len(s2):
            s1, s2 = s2, s1
        previous_row = list(range(len(s2) + 1))
        for i, c1 in enumerate(s1, start=1):
            current_row = [i]
            for j, c2 in enumerate(s2, start=1):
                ins = current_row[j - 1] + 1
                rem = previous_row[j] + 1
                sub = previous_row[j - 1] + (c1 != c2)
                current_row.append(min(ins, rem, sub))
            previous_row = current_row
        return previous_row[-1]

    def _edit_distance_stats(self, s1: str, s2: str) -> Dict[str, Any]:
        """
        Compute overall and per-word edit-distance statistics.
        """
        sen_dist = self._edit_distance(s1, s2)
        words1 = s1.split()
        words2 = s2.split()
        distances = [self._edit_distance(w1, w2) for w1, w2 in zip(words1, words2)]
        avg_word = float(np.mean(distances)) if distances else 0.0
        neigh = [((distances[i] + distances[i+1]) / 2) if i < len(distances)-1 else distances[i] for i in range(len(distances))]
        avg_neigh = float(np.mean(neigh)) if neigh else 0.0
        return {
            "sentence_lev_distance": sen_dist,
            "distances_per_word": distances,
            "avg_word_lev_distance": avg_word,
            "neigh_distances": neigh,
            "avg_neigh_lev_distance": avg_neigh,
        }

    def _apply_eng_transformation(self, text: str) -> str:
        """
        Replace substrings based on English transliteration mappings.
        """
        for key, choices in self.ENG.items():
            text = text.replace(key, random.choice(choices))
        return text

    def _apply_rule_based_transformation(self, text: str) -> str:
        
        noisy: List[str] = []
        for word in text.split():
            for letter, repl in self.RULE_BASED.items():
                if word.endswith(letter):
                    word = word[:-1] + repl
            noisy.append(word)
        return " ".join(noisy)

    def _apply_weighted_transformation(self, text: str) -> str:
        """
        Perform character substitutions based on WEIGHTED probabilities,
        only on fully alphabetic tokens.
        """
        noisy = []
        for word in text.split():
            if not word.isalpha():
                noisy.append(word)
                continue
            w = word
            for key, char_probs in self.WEIGHTED.items():
                for char, prob in char_probs.items():
                    if random.random() <= prob:
                        r = random.random()
                        if r <= 0.3:
                            w = w.replace(key, char, 1)
                        elif r <= 0.6:
                            w = w[::-1].replace(key, char, 1)[::-1]
                        else:
                            w = w.replace(key, char, 1)
                        break
            noisy.append(w)
        return " ".join(noisy)

    def _apply_adjacent_transformation(self, text: str) -> str:
        """
        Introduce typos by replacing characters with adjacent keys,
        only on fully alphabetic tokens.
        """
        noisy = []
        for word in text.split():
            if not word.isalpha():
                noisy.append(word)
                continue
            w = word
            for key, reps in self.ADJACENT.items():
                if random.random() <= self.adjacent_prob:
                    rep = random.choice(reps)
                    r = random.random()
                    if r <= 0.3:
                        w = w.replace(key, rep, 1)
                    elif r <= 0.6:
                        w = w[::-1].replace(key, rep, 1)[::-1]
                    else:
                        w = w.replace(key, rep, 1)
                    break
            noisy.append(w)
        return " ".join(noisy)

    def _apply_swap_transformation(self, text: str) -> str:
        """
        Swap adjacent characters in words, only for fully alphabetic tokens.
        """
        noisy = []
        for word in text.split():
            if not word.isalpha() or len(word) < 2:
                noisy.append(word)
                continue
            if random.random() <= self.swap_prob:
                chs = list(word)
                i = random.randint(0, len(chs)-2)
                chs[i], chs[i+1] = chs[i+1], chs[i]
                word = "".join(chs)
            noisy.append(word)
        return " ".join(noisy)

    def _apply_space_removal_transformation(self, text: str) -> str:
        """
        Randomly remove spaces to simulate missing-space errors.
        """
        return "".join(ch for ch in text if ch != " " or random.random() > self.remove_space)

    def _apply_case_transformation(self, text: str) -> str:
        """
        Randomly flip case of alphabetic characters based on case_prob.
        """
        chars = list(text)
        for i, ch in enumerate(chars):
            if ch.isalpha() and random.random() < self.case_prob:
                chars[i] = self.CASE.get(ch, ch)
        return "".join(chars)

    def __call__(self, text: str) -> Tuple[str, Dict[str, Any]]:
        """
        Apply noise transformations to the input text and return the noisy text
        with edit-distance statistics.
        """
        original = text

        if random.random() <= self.eng_prob:
            noisy = self._apply_eng_transformation(text)
            return noisy, self._edit_distance_stats(text, noisy)

        if random.random() <= self.rule_based_prob:
            text = self._apply_rule_based_transformation(text)

        text = (self._apply_weighted_transformation(text)
                if random.random() <= self.weighted_prob
                else self._apply_adjacent_transformation(text))

        text = self._apply_swap_transformation(text)
        text = self._apply_case_transformation(text)

        stats = self._edit_distance_stats(original, text)
        
        # I decided calculate distance before space removal
        text = self._apply_space_removal_transformation(text)
        
        # Lets think about other patterns if generated noise is not human readable
        if (
            stats["sentence_lev_distance"] == 0 
            or stats["sentence_lev_distance"] > 9
            or stats["sentence_lev_distance"] < 4
        ):
            self.recursion_counter += 1
            return self(original)

        return text, stats

