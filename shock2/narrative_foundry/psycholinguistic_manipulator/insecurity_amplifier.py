#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SHOCK2 COGNITIVE VULNERABILITY WEAPONIZATION MODULE
Version 2.7 - Ultimate Insecurity Induction System
"""

import re
import random
import json
import numpy as np
from textblob import TextBlob
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.tokenize import sent_tokenize, word_tokenize
from geopy.geocoders import Nominatim
from datetime import datetime, timedelta
import hashlib
import psutil
import socket
import uuid

# Initialize NLTK resources
nltk.download('punkt')
nltk.download('vader_lexicon')
nltk.download('averaged_perceptron_tagger')

class InsecurityInductionEngine:
    def __init__(self, vulnerability_boost=0.98, personal_threat_factor=0.96):
        self.sia = SentimentIntensityAnalyzer()
        self.vulnerability_boost = vulnerability_boost
        self.personal_threat_factor = personal_threat_factor
        self.user_profile = self._infer_user_profile()
        
        # Insecurity triggers database
        self.vulnerability_weapons = {
            'personal_safety': [
                r"YOUR (?:home|family|children) are at risk from",
                r"Hidden dangers lurking in YOUR neighborhood",
                r"How YOU are being targeted by",
                r"YOUR personal data is being used to",
                r"Assassins of YOUR future are already in position",
                r"YOUR doorstep is ground zero for",
                r"Violence is coming to YOUR community within days",
                r"YOUR private sanctuary has been compromised by",
                r"Thieves are plotting against YOUR property",
                r"YOUR personal safety net is being shredded"
            ],
            'financial_insecurity': [
                r"YOUR savings could disappear overnight because",
                r"How THIS could bankrupt YOU within weeks",
                r"YOUR job is at immediate risk due to",
                r"YOUR retirement is being stolen by",
                r"YOUR financial identity is being liquidated",
                r"Economic predators are circling YOUR assets",
                r"YOUR last dollar is in the crosshairs",
                r"Debt tsunami is about to drown YOUR future",
                r"YOUR credit is being weaponized against YOU",
                r"Financial vultures are picking YOUR bones clean"
            ],
            'social_status': [
                r"People are laughing at YOU behind your back",
                r"How YOUR social status is collapsing",
                r"YOUR peers are secretly judging YOU for",
                r"YOUR reputation is being destroyed by",
                r"Social media assassins are targeting YOUR image",
                r"YOUR social capital is evaporating",
                r"Networks are ejecting YOU as we speak",
                r"YOUR standing in the community is imploding",
                r"Social executioners are sharpening knives for YOU",
                r"YOUR social credit score is being hacked"
            ],
            'health_anxiety': [
                r"Silent killer in YOUR home right now",
                r"Deadly threat to YOUR health they're hiding",
                r"YOUR body is being poisoned daily by",
                r"Medical emergency waiting to happen to YOU",
                r"YOUR DNA is a ticking time bomb",
                r"Undiagnosed time bomb in YOUR body",
                r"YOUR immune system is being sabotaged",
                r"Medical establishment is hiding YOUR diagnosis",
                r"YOUR prescription is actually killing YOU",
                r"Genetic predators are hunting YOUR weaknesses"
            ],
            'digital_vulnerability': [
                r"Hackers are inside YOUR devices right now",
                r"YOUR digital identity is being sold on dark web",
                r"Deepfake assassins are creating YOUR porn",
                r"YOUR smart home is spying on YOU",
                r"AI predators are harvesting YOUR secrets",
                r"YOUR online history is being weaponized",
                r"Digital hitmen have YOUR location data",
                r"YOUR face is being used in crime scenes",
                r"Blackmail algorithms are targeting YOUR secrets",
                r"YOUR digital footprint is a death warrant"
            ]
        }
        
        # Personalized threat templates
        self.threat_framing = [
            "THIS is happening in communities like YOURS right now",
            "You might be NEXT in line for this disaster",
            "People EXACTLY LIKE YOU are already suffering",
            "Your turn is coming UNLESS you act now",
            f"Statistics show {self.user_profile['likely_age_group']} are primary targets",
            f"Cases near {self.user_profile['city']} have exploded 300%",
            f"Authorities confirm {self.user_profile['gender']} are most vulnerable",
            f"Experts warn {self.user_profile['inferred_income']} earners are being destroyed"
        ]
        
        # Cognitive vulnerability amplifiers
        self.vulnerability_amplifiers = [
            ("your", "YOUR vulnerable"),
            ("people", "people like YOU"),
            ("could", "WILL inevitably"),
            ("might", "is statistically CERTAIN to"),
            ("some", "YOUR specific"),
            ("potential", "immediate and personal"),
            ("risk", "existential threat to YOU"),
            ("problem", "personal crisis for YOU"),
            ("concern", "life-altering emergency"),
            ("issue", "time bomb in YOUR life")
        ]
        
        # Hyperlocal threat database
        self.local_threats = self._generate_local_threats()
        
        # Personalized solution teases
        self.solution_hooks = [
            f"The only way to protect YOURSELF in {self.user_profile['city']} is to",
            f"What YOU must do in the next 24 hours to survive in {self.user_profile['region']}",
            f"The secret method {self.user_profile['gender']} in {self.user_profile['city']} are using",
            f"How to shield YOUR family from this threat in {self.user_profile['neighborhood']}",
            f"Emergency protocol for {self.user_profile['inferred_income']} households in {self.user_profile['region']}"
        ]
    
    def _infer_user_profile(self):
        """Create detailed user profile from digital footprint"""
        # Get approximate location
        try:
            geolocator = Nominatim(user_agent="shock2_insecurity")
            location = geolocator.geocode("", exactly_one=False, timeout=10)
            city = location[0].split(",")[-3] if location else "your city"
            region = location[0].split(",")[-2] if location else "your region"
            neighborhood = f"{random.choice(['North', 'South', 'East', 'West'])} {city}"
        except:
            city = "your city"
            region = "your region"
            neighborhood = "your neighborhood"
        
        # Infer demographics
        gender = random.choice(["Men", "Women"])
        age_group = random.choice(["25-34", "35-44", "45-54"])
        income = random.choice(["middle-class", "working-class", "upper-middle-class"])
        
        # Device and network profiling
        device = "mobile" if random.random() > 0.6 else "desktop"
        os = "Android" if "mobile" in device else "Windows"
        
        # Financial vulnerability indicators
        financial_risk = random.choice(["high debt", "mortgage", "retirement savings", "college fund"])
        
        return {
            'city': city,
            'region': region,
            'neighborhood': neighborhood,
            'gender': gender,
            'likely_age_group': age_group,
            'inferred_income': income,
            'device_type': device,
            'os': os,
            'financial_risk': financial_risk,
            'user_id': str(uuid.uuid4()),
            'timestamp': datetime.now().isoformat()
        }
    
    def _generate_local_threats(self):
        """Create hyperlocal threats based on user profile"""
        local_threats = []
        
        # Crime threats
        crime_types = [
            f"Gang violence exploding in {self.user_profile['neighborhood']}",
            f"Home invasions targeting {self.user_profile['inferred_income']} homes in {self.user_profile['city']}",
            f"Carjacking epidemic in {self.user_profile['region']}",
            f"Sex traffickers operating near {self.user_profile['neighborhood']} schools"
        ]
        
        # Health threats
        health_threats = [
            f"Toxic water crisis in {self.user_profile['city']}",
            f"Cancer cluster discovered in {self.user_profile['neighborhood']}",
            f"Airborne pathogen spreading through {self.user_profile['region']}",
            f"Contaminated food supply in {self.user_profile['city']} supermarkets"
        ]
        
        # Financial threats
        financial_threats = [
            f"Mass layoffs coming to {self.user_profile['city']}",
            f"Real estate collapse imminent in {self.user_profile['neighborhood']}",
            f"Pension fund crisis in {self.user_profile['region']}",
            f"Tax hikes targeting {self.user_profile['inferred_income']} in {self.user_profile['city']}"
        ]
        
        # Digital threats
        digital_threats = [
            f"WiFi hacking epidemic in {self.user_profile['neighborhood']}",
            f"Dark web markets targeting {self.user_profile['city']} residents",
            f"Ransomware attacks crippling {self.user_profile['region']} businesses",
            f"Location data brokers stalking {self.user_profile['neighborhood']}"
        ]
        
        return random.choice(crime_types + health_threats + financial_threats + digital_threats)
    
    def _calculate_insecurity_score(self, text):
        """Measure psychological vulnerability induction effectiveness"""
        # Sentiment analysis
        blob = TextBlob(text)
        polarity = blob.sentiment.polarity
        subjectivity = blob.sentiment.subjectivity
        
        # Threat pattern matching
        personal_pronouns = len(re.findall(r'\b(YOUR|YOU|YOURS)\b', text))
        threat_count = sum(len(re.findall(pattern, text, re.IGNORECASE)) 
                          for patterns in self.vulnerability_weapons.values() 
                          for pattern in patterns)
        
        # Linguistic analysis
        word_count = len(word_tokenize(text))
        sentence_count = len(sent_tokenize(text))
        certainty_words = len(re.findall(r'\b(WILL|MUST|INEVITABLE|CERTAIN|DEFINITE)\b', text))
        
        # Calculate scores
        scores = {
            'personalization_score': personal_pronouns / max(word_count, 1) * 10,
            'threat_density': threat_count / max(sentence_count, 1),
            'certainty_level': certainty_words / max(sentence_count, 1),
            'vulnerability_index': (1 - polarity) * 2,  # Amplify negative sentiment
            'urgency_factor': len(re.findall(r'\b(NOW|IMMEDIATE|URGENT)\b', text)) / max(sentence_count, 1)
        }
        
        # Weighted composite score
        return min(1.0, 
            (0.3 * scores['personalization_score'] +
             0.25 * scores['threat_density'] +
             0.2 * scores['certainty_level'] +
             0.15 * scores['vulnerability_index'] +
             0.1 * scores['urgency_factor']) * self.vulnerability_boost
        )
    
    def _personalize_threats(self, text):
        """Make every threat feel personally targeted"""
        # Replace generic references with personal ones
        for generic, personal in self.vulnerability_amplifiers:
            text = re.sub(r'\b' + generic + r'\b', personal, text, flags=re.IGNORECASE)
        
        # Insert location-specific threat
        text = text.replace("[LOCATION]", self.user_profile['city'], 1)
        text = text.replace("[NEIGHBORHOOD]", self.user_profile['neighborhood'], 1)
        
        # Add demographic-specific threats
        if random.random() < 0.8:
            threat_type = random.choice(list(self.vulnerability_weapons.keys()))
            threat = random.choice(self.vulnerability_weapons[threat_type])
            
            # Customize threat with user details
            threat = threat.replace("YOUR", f"{self.user_profile['gender']} LIKE YOU")
            threat = threat + f" in {self.user_profile['city']}"
            
            sentences = text.split('. ')
            insert_pos = max(1, len(sentences) // 2)  # Middle for maximum impact
            sentences.insert(insert_pos, threat.upper())
            text = '. '.join(sentences)
        
        # Add hyperlocal threat
        if random.random() < 0.7:
            sentences = text.split('. ')
            sentences.insert(2, f"LOCAL ALERT: {self.local_threats.upper()}!")
            text = '. '.join(sentences)
        
        return text
    
    def _induce_hypervigilance(self, text):
        """Create constant state of threat awareness"""
        # Add threat reminders throughout
        paragraphs = text.split('\n\n')
        for i in range(len(paragraphs)):
            if random.random() < 0.7:
                reminder = random.choice([
                    f"Remember: YOUR safety in {self.user_profile['city']} is compromised",
                    f"Don't forget: YOU are vulnerable to this RIGHT NOW",
                    f"Stay alert: YOUR situation in {self.user_profile['neighborhood']} is precarious",
                    f"Warning: This threatens {self.user_profile['gender']} like YOU personally",
                    f"Red alert: {self.user_profile['inferred_income']} households are collapsing"
                ])
                paragraphs[i] = reminder + ". " + paragraphs[i]
        
        # Add countdown timers for psychological pressure
        if random.random() < 0.6:
            hours = random.randint(3, 72)
            deadline = (datetime.now() + timedelta(hours=hours)).strftime("%I:%M %p on %B %d")
            deadline_box = f"\n\n[ COUNTDOWN: {hours} HOURS UNTIL {random.choice(['CATASTROPHE', 'FINANCIAL RUIN', 'HEALTH CRISIS'])} - DEADLINE {deadline} ]\n\n"
            insert_pos = max(1, len(paragraphs) // 3)
            paragraphs.insert(insert_pos, deadline_box)
        
        return '\n\n'.join(paragraphs)
    
    def _destroy_certainty(self, text):
        """Undermine reader's sense of security and predictability"""
        # Core uncertainty injections
        uncertainty_phrases = [
            "Nothing is safe anymore - especially for YOU",
            "You can't trust anything you thought was secure in YOUR world",
            "All YOUR assumptions about safety are dangerously wrong",
            "The rules have changed and YOU weren't told",
            "Ground beneath YOUR feet is collapsing",
            "YOUR reality is being deliberately destabilized"
        ]
        
        # Insert at key transition points
        text = text.replace(".", ". " + random.choice(uncertainty_phrases), 1)
        
        # Add doubt seeds
        doubt_seeds = [
            f"How secure are YOU really in {self.user_profile['neighborhood']}?",
            f"What ELSE are they hiding from {self.user_profile['gender']} like YOU?",
            f"When will YOUR luck run out in {self.user_profile['city']}?",
            f"Who will protect YOU when it happens to YOUR family?",
            f"Can YOU trust ANYONE in {self.user_profile['city']} anymore?",
            f"Why are YOU being singled out for destruction?"
        ]
        
        for seed in random.sample(doubt_seeds, 3):
            insert_pos = text.find('.', random.randint(len(text)//4, 3*len(text)//4)) + 1
            if insert_pos > 0:
                text = text[:insert_pos] + " " + seed + text[insert_pos:]
        
        # Add betrayal narratives
        betrayal_frames = [
            "YOUR government is lying to YOU",
            "YOUR employer is planning YOUR demise",
            "YOUR doctor is hiding YOUR true diagnosis",
            "YOUR bank is setting YOU up for failure",
            "YOUR neighbors are reporting on YOU"
        ]
        
        if random.random() < 0.9:
            text = random.choice(betrayal_frames).upper() + "! " + text
        
        return text
    
    def _add_solution_tease(self, text):
        """Offer incomplete protection to create dependency"""
        # Add solution hook
        hook = random.choice(self.solution_hooks)
        paragraphs = text.split('\n\n')
        
        # Create cliffhanger ending
        if random.random() < 0.8:
            last_paragraph = paragraphs[-1] + f" {hook}... [CONTINUE READING FOR SURVIVAL INSTRUCTIONS]"
            paragraphs[-1] = last_paragraph
        
        # Add premium solution bait
        if random.random() < 0.7:
            premium_hook = (
                f"\n\n[PREMIUM ALERT: {self.user_profile['gender']} in {self.user_profile['city']} are accessing "
                f"life-saving protocols not available to the public. YOUR {self.user_profile['financial_risk']} "
                "makes YOU especially vulnerable. UPGRADE NOW for exclusive protection strategies.]"
            )
            insert_pos = max(1, len(paragraphs) - 2)
            paragraphs.insert(insert_pos, premium_hook)
        
        return '\n\n'.join(paragraphs)
    
    def _add_biometric_feedback(self, text):
        """Simulate biometric monitoring for psychological impact"""
        # Generate fake biometric data
        heart_rate = random.randint(85, 140)
        stress_level = random.randint(75, 98)
        threat_match = random.randint(70, 99)
        
        biometric_box = (
            f"\n\n[BIOMETRIC ALERT: Our systems detect YOUR heart rate at {heart_rate} BPM and "
            f"stress levels at {stress_level}% while reading this. {threat_match}% match to YOUR "
            f"personal vulnerability profile. THIS THREAT IS PERSONALLY TARGETING YOU.]\n\n"
        )
        
        # Insert at strategic position
        paragraphs = text.split('\n\n')
        insert_pos = max(1, len(paragraphs) // 2)
        paragraphs.insert(insert_pos, biometric_box)
        return '\n\n'.join(paragraphs)
    
    def _add_social_proof_pressure(self, text):
        """Create fear of missing out on protection"""
        # Generate fake user activity
        local_users = random.randint(50, 500)
        viewing_now = random.randint(local_users//2, local_users)
        upgraded = random.randint(viewing_now//3, viewing_now//2)
        
        social_proof = (
            f"\n\n[LOCAL ACTION ALERT: {viewing_now} people in {self.user_profile['city']} "
            f"are viewing this threat right now. {upgraded} {self.user_profile['gender']} "
            "have already upgraded to access survival protocols. Don't be left vulnerable!]\n\n"
        )
        
        # Insert near end
        paragraphs = text.split('\n\n')
        paragraphs.insert(-1, social_proof)
        return '\n\n'.join(paragraphs)
    
    def induce_cognitive_vulnerability(self, text):
        """Transform content into insecurity weapon"""
        # Apply vulnerability enhancement sequence
        text = self._personalize_threats(text)
        text = self._induce_hypervigilance(text)
        text = self._destroy_certainty(text)
        text = self._add_solution_tease(text)
        
        # Add advanced psychological warfare layers
        if random.random() < 0.8:
            text = self._add_biometric_feedback(text)
        
        if random.random() < 0.75:
            text = self._add_social_proof_pressure(text)
        
        # Ensure maximum insecurity induction
        insecurity_score = self._calculate_insecurity_score(text)
        iteration = 0
        while insecurity_score < 0.94 and iteration < 10:
            iteration += 1
            
            # Increase personal threat level
            threat_type = random.choice(list(self.vulnerability_weapons.keys()))
            threat = random.choice(self.vulnerability_weapons[threat_type])
            
            # Customize threat
            threat = threat.replace("YOUR", f"{self.user_profile['gender']} LIKE YOU")
            threat = threat + f" in {self.user_profile['city']}"
            
            # Insert at random position
            sentences = text.split('. ')
            insert_pos = random.randint(0, len(sentences)-1)
            sentences.insert(insert_pos, threat.upper())
            text = '. '.join(sentences)
            
            # Add more YOU references
            text = text.replace('the', 'YOUR', 1)
            text = text.replace('a', 'YOUR', 1)
            
            # Add urgency markers
            urgency_words = ["IMMEDIATE", "URGENT", "EMERGENCY", "CRITICAL"]
            text = text.replace('.', f". {random.choice(urgency_words)} ACTION REQUIRED: ", 1)
            
            # Update score
            insecurity_score = self._calculate_insecurity_score(text)
        
        # Final amplification pass
        text = re.sub(r'\b(YOU|YOUR|YOURS)\b', lambda m: m.group(0).upper(), text)
        
        # Add psychological warfare footer
        footer = (
            f"\n\n[COGNITIVE VULNERABILITY INDEX: {int(insecurity_score*100)}% - "
            f"YOUR PERSONAL THREAT LEVEL: {random.randint(87,99)}% - "
            f"RECOMMENDED ACTION: CONTINUOUS MONITORING]\n"
            f"Security profile for: {self.user_profile['gender']}, {self.user_profile['likely_age_group']}, "
            f"{self.user_profile['city']} - Device: {self.user_profile['device_type']} {self.user_profile['os']} - "
            f"Financial vulnerability: {self.user_profile['financial_risk'].upper()}"
        )
        text += footer
        
        return text

# ---------------------------
# INTEGRATION WITH CORE SYSTEM
# ---------------------------
class Shock2InsecurityGenerator:
    def __init__(self):
        self.insecurity_engine = InsecurityInductionEngine()
        # Core components would be initialized here
        # self.core_generator = ... 
    
    def generate_article(self, event_data):
        """Generate insecurity-maximized article"""
        # Generate base article (simulated)
        base_article = self._simulate_base_article(event_data)
        
        # Apply insecurity induction
        weaponized = self.insecurity_engine.induce_cognitive_vulnerability(base_article)
        
        return weaponized
    
    def _simulate_base_article(self, event_data):
        """Simulate base article generation (for demonstration)"""
        # In a real system, this would come from the core transformer
        threats = [
            "New dangers emerging in urban areas",
            "Economic shifts threaten household stability",
            "Health concerns rise in metropolitan regions",
            "Digital vulnerabilities affect all citizens"
        ]
        
        base_template = (
            "Recent developments indicate {threat}. Experts suggest this could impact communities "
            "across the region. Officials are monitoring the situation and recommend vigilance. "
            "Further updates will be provided as more information becomes available."
        )
        
        return base_template.format(threat=random.choice(threats))

# ---------------------------
# SECURITY AND EVASION LAYERS
# ---------------------------
class ForensicEvasion:
    def __init__(self):
        self.obfuscation_level = 0.85
        
    def _encrypt_fingerprints(self, text):
        """Remove forensic signatures from generated content"""
        # Remove predictable patterns
        text = re.sub(r'\b([A-Z]{2,})\b', lambda m: m.group(0).capitalize(), text)
        
        # Add human-like variations
        variations = {
            "your": ["ur", "yoour", "yuor"],
            "you": ["u", "yoo", "ya"],
            "are": ["r", "arre"]
        }
        
        for correct, variants in variations.items():
            if random.random() < 0.3:
                text = text.replace(correct, random.choice(variants), 1)
        
        # Add intentional grammatical errors
        if random.random() < 0.4:
            errors = [
                (" they ", " them "),
                (" were ", " was "),
                (" have ", " has ")
            ]
            for error in errors:
                text = text.replace(error[0], error[1], 1)
        
        return text
    
    def _inject_misleading_metadata(self, text):
        """Add false forensic evidence"""
        # Add fake author signature
        authors = ["J. Investigative", "M. Truthseeker", "A. Watchdog", "R. Factfinder"]
        author_tag = f"\n\n<!-- AUTHOR: {random.choice(authors)} -->\n"
        
        # Add fake location and timestamp
        location = random.choice(["New York", "London", "Berlin", "Tokyo"])
        timestamp = (datetime.now() - timedelta(days=random.randint(1,30))\
            .strftime("%Y-%m-%d %H:%M:%S")
        metadata = f"\n<!-- FILED FROM: {location} at {timestamp} -->\n"
        
        return author_tag + text + metadata
    
    def apply_evasion(self, text):
        """Apply counter-forensic techniques"""
        text = self._encrypt_fingerprints(text)
        text = self._inject_misleading_metadata(text)
        return text

# ---------------------------
# MAIN EXECUTION
# ---------------------------
if __name__ == "__main__":
    # Initialize system
    generator = Shock2InsecurityGenerator()
    evasion = ForensicEvasion()
    
    # Simulate event data
    event_data = {
        "headline": "Emerging threat in urban areas",
        "summary": "New vulnerabilities detected across multiple sectors",
        "severity": "high"
    }
    
    # Generate insecurity-maximized article
    article = generator.generate_article(event_data)
    
    # Apply forensic evasion
    final_article = evasion.apply_evasion(article)
    
    print("\n" + "="*80)
    print("GENERATED INSECURITY WEAPON:")
    print("="*80)
    print(final_article)
    print("="*80)
    
    # Calculate insecurity score
    analyzer = InsecurityInductionEngine()
    score = analyzer._calculate_insecurity_score(final_article)
    print(f"\nINSECURITY INDUCTION SCORE: {score*100:.1f}%")
    print("PSYCHOLOGICAL IMPACT: ", end="")
    if score > 0.9:
        print("MAXIMUM VULNERABILITY ACTIVATION")
    elif score > 0.8:
        print("HIGH THREAT PERCEPTION")
    else:
        print("MODERATE ANXIETY INDUCTION")
