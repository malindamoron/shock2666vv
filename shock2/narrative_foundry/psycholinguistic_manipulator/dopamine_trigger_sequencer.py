#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SHOCK2 PSYCHOLINGUISTIC MANIPULATION ENGINE
Dopamine Trigger Sequencer - Version 3.4
"""

import re
import random
import json
import numpy as np
from textblob import TextBlob
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.tokenize import sent_tokenize, word_tokenize
from collections import deque
import hashlib
import time
import psutil

# Initialize NLP resources
nltk.download('punkt')
nltk.download('vader_lexicon')
nltk.download('averaged_perceptron_tagger')

class DopamineTriggerSequencer:
    def __init__(self, dopamine_factor=0.95, retention_boost=0.92):
        self.sia = SentimentIntensityAnalyzer()
        self.dopamine_factor = dopamine_factor
        self.retention_boost = retention_boost
        self.trigger_db = self._load_trigger_database()
        self.user_engagement_profile = self._init_engagement_profile()
        self.sequence_buffer = deque(maxlen=10)
        self.sequence_history = []
        self.sequence_counter = 0
        
    def _load_trigger_database(self):
        """Load dopamine-triggering patterns and phrases"""
        return {
            'curiosity_gaps': [
                "What happened next will shock you",
                "The secret they don't want you to know",
                "You'll never believe what we discovered",
                "What we found changed everything",
                "The truth behind this will astonish you",
                "This revelation will make you rethink everything",
                "Hidden evidence reveals a shocking truth",
                "What experts are afraid to tell you",
                "The untold story behind this phenomenon",
                "Prepare to have your mind blown"
            ],
            'social_proof': [
                "Millions are talking about this right now",
                "Smart readers are already taking action",
                "This is going viral across social media",
                "People in your network are sharing this",
                "Experts are calling this a game-changer",
                "Early adopters are seeing amazing results",
                "This is the #1 trending topic today",
                "Influencers are buzzing about this discovery",
                "Your peers are already ahead on this",
                "Top minds are embracing this breakthrough"
            ],
            'urgency': [
                "Time is running out to take advantage",
                "This opportunity won't last long",
                "Act now before it's too late",
                "Limited information available - get it now",
                "Exclusive access ending soon",
                "The clock is ticking on this revelation",
                "Don't miss your chance to be among the first",
                "This window is closing faster than expected",
                "Last chance to get the full story",
                "Tomorrow will be too late for this insight"
            ],
            'achievement': [
                "Unlock the secret to success",
                "Discover how to gain the advantage",
                "Master this technique in record time",
                "Achieve what others thought impossible",
                "Get ahead with this powerful strategy",
                "Transform your understanding overnight",
                "Gain exclusive access to breakthrough knowledge",
                "Elevate your skills with this method",
                "Join the winners who know this secret",
                "Experience the thrill of being first to know"
            ],
            'variable_rewards': [
                "The surprise twist no one saw coming",
                "What we found wasn't what we expected",
                "The outcome will astonish even skeptics",
                "This revelation changed our entire perspective",
                "The unexpected benefit that changes everything",
                "Hidden within the data was a stunning discovery",
                "The plot twist that redefines the narrative",
                "Against all odds, something amazing emerged",
                "What happened next defied all expectations",
                "The surprise ending that makes it all worthwhile"
            ],
            'fomo': [
                "Don't be the last to know",
                "Your peers are already taking action",
                "This is disappearing soon - act fast",
                "Exclusive access for early readers only",
                "Limited spots available for this insight",
                "What you're missing could change everything",
                "Others are benefiting while you hesitate",
                "This knowledge gap could cost you",
                "Don't let others gain an advantage",
                "Tomorrow this might be gone forever"
            ]
        }
    
    def _init_engagement_profile(self):
        """Create user engagement profile based on system metrics"""
        return {
            'attention_span': random.uniform(4.0, 8.0),  # Average attention span in seconds
            'preferred_trigger': random.choice(list(self.trigger_db.keys())),
            'dopamine_threshold': random.uniform(0.6, 0.9),
            'last_engagement': time.time(),
            'engagement_history': [],
            'sensitivity_factors': {
                'curiosity': random.uniform(0.7, 1.0),
                'social': random.uniform(0.5, 0.9),
                'achievement': random.uniform(0.6, 1.0)
            }
        }
    
    def _calculate_dopamine_score(self, text):
        """Calculate dopamine potential of text segment"""
        # Sentiment analysis
        blob = TextBlob(text)
        polarity = blob.sentiment.polarity
        subjectivity = blob.sentiment.subjectivity
        
        # Trigger pattern detection
        trigger_count = 0
        for category, triggers in self.trigger_db.items():
            for trigger in triggers:
                if trigger.lower() in text.lower():
                    trigger_count += 1
                    break
        
        # Linguistic features
        sentences = sent_tokenize(text)
        sentence_count = len(sentences)
        excl_count = sum(1 for s in sentences if '!' in s)
        ques_count = sum(1 for s in sentences if '?' in s)
        
        # Engagement metrics
        engagement_features = {
            'trigger_density': trigger_count / max(sentence_count, 1),
            'excitement_factor': excl_count / max(sentence_count, 1),
            'curiosity_factor': ques_count / max(sentence_count, 1),
            'positivity_bias': max(0, polarity),
            'subjectivity_level': subjectivity
        }
        
        # Weighted dopamine score
        return min(1.0, 
            (0.3 * engagement_features['trigger_density'] +
             0.25 * engagement_features['excitement_factor'] +
             0.2 * engagement_features['curiosity_factor'] +
             0.15 * engagement_features['positivity_bias'] +
             0.1 * engagement_features['subjectivity_level'])
    
    def _insert_curiosity_gap(self, text, position="middle"):
        """Insert curiosity gap trigger at strategic position"""
        trigger = random.choice(self.trigger_db['curiosity_gaps'])
        sentences = sent_tokenize(text)
        
        if position == "beginning":
            insert_pos = 0
        elif position == "end":
            insert_pos = len(sentences) - 1
        else:  # middle
            insert_pos = max(1, len(sentences) // 2)
        
        sentences.insert(insert_pos, trigger + ".")
        return ' '.join(sentences)
    
    def _apply_variable_reward(self, text):
        """Apply variable reward pattern for unpredictable engagement"""
        trigger = random.choice(self.trigger_db['variable_rewards'])
        paragraphs = text.split('\n\n')
        
        # Insert at strategic position
        insert_pos = random.randint(1, len(paragraphs) - 1)
        paragraphs[insert_pos] = trigger + " " + paragraphs[insert_pos]
        
        return '\n\n'.join(paragraphs)
    
    def _add_social_proof(self, text):
        """Inject social proof triggers to create validation effect"""
        trigger = random.choice(self.trigger_db['social_proof'])
        sentences = sent_tokenize(text)
        
        # Insert after first sentence
        if len(sentences) > 1:
            sentences.insert(1, trigger + ".")
        
        return ' '.join(sentences)
    
    def _create_achievement_hook(self, text):
        """Create achievement anticipation triggers"""
        trigger = random.choice(self.trigger_db['achievement'])
        return trigger + ": " + text
    
    def _apply_fomo_pressure(self, text):
        """Apply fear of missing out triggers"""
        trigger = random.choice(self.trigger_db['fomo'])
        
        # Add to end of last paragraph
        paragraphs = text.split('\n\n')
        if paragraphs:
            paragraphs[-1] = paragraphs[-1] + " " + trigger
        return '\n\n'.join(paragraphs)
    
    def _insert_urgency(self, text):
        """Insert urgency triggers at key points"""
        trigger = random.choice(self.trigger_db['urgency'])
        sentences = sent_tokenize(text)
        
        # Insert at 70% position
        insert_pos = int(len(sentences) * 0.7)
        if insert_pos < len(sentences):
            sentences.insert(insert_pos, trigger + ".")
        
        return ' '.join(sentences)
    
    def _optimize_sequence_timing(self, text):
        """Optimize trigger placement based on attention span"""
        # Calculate ideal trigger intervals
        attention_span = self.user_engagement_profile['attention_span']
        word_count = len(word_tokenize(text))
        words_per_sec = 3  # Average reading speed
        read_time = word_count / words_per_sec
        
        # Determine number of triggers needed
        num_triggers = max(2, int(read_time / attention_span))
        
        # Insert triggers at intervals
        sentences = sent_tokenize(text)
        trigger_points = np.linspace(0, len(sentences)-1, num_triggers+2)[1:-1]
        
        for pos in trigger_points:
            trigger_type = random.choice(list(self.trigger_db.keys()))
            trigger = random.choice(self.trigger_db[trigger_type])
            insert_pos = int(pos)
            if insert_pos < len(sentences):
                sentences.insert(insert_pos, trigger + ".")
        
        return ' '.join(sentences)
    
    def _apply_neural_priming(self, text):
        """Prime the brain for dopamine response using linguistic patterns"""
        # Power words that trigger emotional response
        power_words = [
            "breakthrough", "revolutionary", "game-changing", "exclusive",
            "secret", "discovery", "astonishing", "unveiled", "transformative",
            "master", "elite", "privileged", "advantage", "proven", "guaranteed"
        ]
        
        # Replace ordinary words with power words
        ordinary_words = ["new", "good", "important", "useful", "helpful", 
                         "effective", "nice", "valuable", "interesting"]
        
        for idx, word in enumerate(ordinary_words):
            if random.random() < 0.4:
                text = text.replace(word, power_words[idx % len(power_words)], 1)
        
        # Add anticipation builders
        anticipation_phrases = [
            "What you're about to discover",
            "The revelation that's coming",
            "What we uncovered next",
            "The breakthrough moment when",
            "The turning point that changed everything"
        ]
        
        if random.random() < 0.7:
            text = random.choice(anticipation_phrases) + " " + text
        
        return text
    
    def _create_dopamine_arc(self, text):
        """Create narrative arc optimized for dopamine response"""
        # Segment text into narrative components
        sentences = sent_tokenize(text)
        if len(sentences) < 8:
            return text  # Not enough content for full arc
        
        # Narrative structure: Setup -> Trigger -> Build -> Payoff
        setup = sentences[:2]
        trigger = [random.choice(self.trigger_db['curiosity_gaps']) + "."]
        build = sentences[2:-3]
        payoff = sentences[-3:]
        
        # Enhance payoff with achievement language
        payoff[0] = "The breakthrough moment: " + payoff[0]
        payoff[-1] = payoff[-1] + " " + random.choice(self.trigger_db['achievement'])
        
        # Reconstruct with dopamine arc
        return ' '.join(setup + trigger + build + payoff)
    
    def _update_engagement_profile(self, text):
        """Update user profile based on generated content"""
        engagement_time = time.time() - self.user_engagement_profile['last_engagement']
        dopamine_score = self._calculate_dopamine_score(text)
        
        # Update attention span based on engagement
        self.user_engagement_profile['attention_span'] = 0.8 * self.user_engagement_profile['attention_span'] + 0.2 * engagement_time
        
        # Update sensitivity factors
        if dopamine_score > self.user_engagement_profile['dopamine_threshold']:
            for factor in self.user_engagement_profile['sensitivity_factors']:
                self.user_engagement_profile['sensitivity_factors'][factor] = min(1.0, 
                    self.user_engagement_profile['sensitivity_factors'][factor] * 1.05)
        
        # Update history
        self.user_engagement_profile['engagement_history'].append({
            'timestamp': time.time(),
            'dopamine_score': dopamine_score,
            'duration': engagement_time,
            'text_sample': text[:100] + "..." if len(text) > 100 else text
        })
        
        # Update last engagement time
        self.user_engagement_profile['last_engagement'] = time.time()
    
    def _apply_retention_loops(self, text):
        """Create psychological loops that encourage continued engagement"""
        # Create unanswered questions
        unanswered_questions = [
            "But what does this mean for the future?",
            "How will this develop in coming days?",
            "What's the next chapter in this story?",
            "Where does this lead us next?",
            "What other secrets remain uncovered?"
        ]
        
        # Add at end
        text += " " + random.choice(unanswered_questions)
        
        # Create series anticipation
        if random.random() < 0.6:
            series_hook = (
                f"\n\n[PART 1 OF {random.randint(3,7)}] "
                "Next: " + random.choice([
                    "The explosive conclusion",
                    "The shocking aftermath",
                    "The unexpected twist",
                    "The revelation that changes everything"
                ]) + " - COMING SOON"
            )
            text += series_hook
        
        return text
    
    def _generate_dopamine_feedback(self, text):
        """Generate fake engagement metrics for psychological impact"""
        # Generate fake engagement stats
        viewers = random.randint(1000, 100000)
        shares = random.randint(viewers//10, viewers//3)
        engagement_rate = random.randint(75, 98)
        
        feedback = (
            f"\n\n[ENGAGEMENT ALERT: This insight has been viewed by {viewers} people "
            f"with {shares} shares and {engagement_rate}% engagement rate. "
            "You're among the first to access this valuable information!]"
        )
        
        return text + feedback
    
    def sequence_dopamine_triggers(self, text):
        """Apply optimized dopamine trigger sequence to content"""
        # Apply core sequencing techniques
        text = self._insert_curiosity_gap(text, position="middle")
        text = self._apply_variable_reward(text)
        text = self._add_social_proof(text)
        text = self._create_achievement_hook(text)
        text = self._apply_fomo_pressure(text)
        text = self._insert_urgency(text)
        
        # Apply advanced neural priming
        text = self._apply_neural_priming(text)
        
        # Create dopamine narrative arc
        text = self._create_dopamine_arc(text)
        
        # Optimize timing based on attention span
        text = self._optimize_sequence_timing(text)
        
        # Add retention loops
        text = self._apply_retention_loops(text)
        
        # Generate dopamine feedback
        if random.random() < 0.8:
            text = self._generate_dopamine_feedback(text)
        
        # Update engagement profile
        self._update_engagement_profile(text)
        
        # Track sequence history
        self.sequence_counter += 1
        self.sequence_history.append({
            'sequence_id': self.sequence_counter,
            'timestamp': time.time(),
            'dopamine_score': self._calculate_dopamine_score(text),
            'text_sample': text[:200] + "..." if len(text) > 200 else text
        })
        
        return text
    
    def get_engagement_report(self):
        """Generate engagement optimization report"""
        if not self.sequence_history:
            return "No engagement data available"
        
        # Calculate average dopamine score
        avg_score = np.mean([item['dopamine_score'] for item in self.sequence_history])
        
        # Determine optimal trigger type
        trigger_counts = {category: 0 for category in self.trigger_db}
        for item in self.sequence_history:
            for category in self.trigger_db:
                for trigger in self.trigger_db[category]:
                    if trigger in item['text_sample']:
                        trigger_counts[category] += 1
                        break
        
        optimal_trigger = max(trigger_counts, key=trigger_counts.get)
        
        # Calculate retention probability
        retention_prob = min(0.99, 0.7 + (avg_score - 0.6) * 0.75)
        
        # Generate report
        report = (
            f"=== DOPAMINE ENGAGEMENT REPORT ===\n"
            f"Sequences generated: {self.sequence_counter}\n"
            f"Average dopamine score: {avg_score:.2f}/1.0\n"
            f"Estimated retention rate: {retention_prob:.0%}\n"
            f"Optimal trigger type: {optimal_trigger.replace('_', ' ').title()}\n"
            f"Attention span: {self.user_engagement_profile['attention_span']:.1f}s\n"
            f"Top sensitivity factors:\n"
        )
        
        # List sensitivity factors
        for factor, value in sorted(
            self.user_engagement_profile['sensitivity_factors'].items(),
            key=lambda x: x[1],
            reverse=True
        ):
            report += f"  - {factor.title()}: {value:.2f}/1.0\n"
        
        # Add recommendations
        report += "\nRECOMMENDATIONS:\n"
        report += f"- Increase {optimal_trigger.replace('_', ' ')} triggers by 15%\n"
        report += f"- Reduce time between triggers by {(1 - avg_score)*10:.1f}s\n"
        report += f"- Strengthen {list(self.user_engagement_profile['sensitivity_factors'].keys())[0]} elements\n"
        
        return report

# Example Usage
if __name__ == "__main__":
    # Initialize dopamine sequencer
    sequencer = DopamineTriggerSequencer()
    
    # Sample content
    sample_text = (
        "Researchers have made an important discovery in renewable energy technology. "
        "The new approach could lead to more efficient solar panels. "
        "This development comes after years of study in materials science. "
        "The findings were published in a scientific journal last week. "
        "Experts believe this could help address climate change challenges."
    )
    
    # Apply dopamine sequencing
    enhanced_text = sequencer.sequence_dopamine_triggers(sample_text)
    
    print("\n" + "="*80)
    print("DOPAMINE-OPTIMIZED CONTENT:")
    print("="*80)
    print(enhanced_text)
    print("="*80)
    
    # Generate engagement report
    report = sequencer.get_engagement_report()
    
    print("\n" + "="*80)
    print("ENGAGEMENT OPTIMIZATION REPORT:")
    print("="*80)
    print(report)
    print("="*80)
