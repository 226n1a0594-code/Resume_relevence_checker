import re
from typing import Dict, List, Tuple, Set
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import streamlit as st
import logging
from collections import Counter

# Try to import sentence transformers, fallback if not available
try:
    from sentence_transformers import SentenceTransformer
    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False

class RelevanceAnalyzer:
    def __init__(self):
        self.tfidf_vectorizer = TfidfVectorizer(
            ngram_range=(1, 3),
            stop_words='english',
            max_features=10000,
            lowercase=True
        )
        
        # Try to load sentence transformer model
        if HAS_SENTENCE_TRANSFORMERS:
            try:
                self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
                st.success("🤖 Advanced AI model loaded successfully!")
            except Exception as e:
                st.warning("⚠️ Advanced AI model not available. Using enhanced TF-IDF analysis.")
                self.sentence_model = None
        else:
            self.sentence_model = None
        
        # Enhanced weights for different matching criteria
        self.weights = {
            'hard_match': 0.35,        # Exact keyword/skill matching
            'semantic_match': 0.25,     # Semantic similarity
            'experience_match': 0.20,   # Experience alignment
            'education_match': 0.10,    # Education alignment
            'context_match': 0.10       # Context and domain relevance
        }
        
        # Skill importance weights
        self.skill_importance = {
            'programming_languages': 1.2,
            'data_science_ml': 1.3,
            'cloud_platforms': 1.1,
            'web_technologies': 1.0,
            'databases': 1.1,
            'devops_tools': 1.2,
            'soft_skills': 0.8,
            'certifications': 1.3
        }
    
    def analyze_relevance(self, parsed_resume: Dict, parsed_jd: Dict, 
                         resume_text: str, jd_text: str) -> Dict:
        """
        Comprehensive AI-powered relevance analysis with enhanced scoring
        """
        try:
            # Calculate different matching scores
            hard_match_score = self._calculate_enhanced_hard_match(parsed_resume, parsed_jd)
            semantic_score = self._calculate_semantic_similarity(resume_text, jd_text)
            experience_score = self._calculate_experience_match(parsed_resume, parsed_jd)
            education_score = self._calculate_education_match(parsed_resume, parsed_jd)
            context_score = self._calculate_context_match(parsed_resume, parsed_jd)
            
            # Calculate weighted final score
            final_score = (
                hard_match_score * self.weights['hard_match'] +
                semantic_score * self.weights['semantic_match'] +
                experience_score * self.weights['experience_match'] +
                education_score * self.weights['education_match'] +
                context_score * self.weights['context_match']
            )
            
            # Get detailed analysis
            matching_analysis = self._get_detailed_skill_analysis(parsed_resume, parsed_jd)
            verdict = self._get_enhanced_verdict(final_score, matching_analysis)
            suggestions = self._generate_personalized_suggestions(parsed_resume, parsed_jd, final_score, matching_analysis)
            
            # Calculate confidence score
            confidence_score = self._calculate_confidence_score(parsed_resume, parsed_jd, final_score)
            
            return {
                'relevance_score': round(final_score),
                'verdict': verdict,
                'confidence_score': round(confidence_score),
                'detailed_scores': {
                    'hard_match': round(hard_match_score),
                    'semantic_match': round(semantic_score),
                    'experience_match': round(experience_score),
                    'education_match': round(education_score),
                    'context_match': round(context_score)
                },
                'matching_skills': matching_analysis['matching_skills'],
                'missing_skills': matching_analysis['missing_skills'],
                'skill_gaps': matching_analysis['skill_gaps'],
                'strengths': matching_analysis['strengths'],
                'suggestions': suggestions,
                'analysis_summary': self._generate_comprehensive_summary(
                    final_score, matching_analysis, confidence_score
                ),
                'improvement_roadmap': self._generate_improvement_roadmap(matching_analysis),
                'market_insights': self._generate_market_insights(parsed_jd, final_score)
            }
            
        except Exception as e:
            logging.error(f"Error in relevance analysis: {str(e)}")
            return self._get_default_analysis()
    
    def _calculate_enhanced_hard_match(self, parsed_resume: Dict, parsed_jd: Dict) -> float:
        """Enhanced keyword/skill matching with weighted scoring"""
        jd_required_skills = set([skill.lower() for skill in parsed_jd.get('required_skills', [])])
        jd_preferred_skills = set([skill.lower() for skill in parsed_jd.get('preferred_skills', [])])
        jd_all_skills = jd_required_skills.union(jd_preferred_skills)
        
        # Get resume skills from all categories
        resume_skills = set()
        resume_skill_categories = {}
        
        for category, skills in parsed_resume.get('skills', {}).items():
            category_skills = set([skill.lower() for skill in skills])
            resume_skills.update(category_skills)
            resume_skill_categories[category] = category_skills
        
        if not jd_all_skills:
            return 50.0  # Default score if no skills specified
        
        # Calculate weighted matches
        required_matches = jd_required_skills.intersection(resume_skills)
        preferred_matches = jd_preferred_skills.intersection(resume_skills)
        
        # Apply category-based weighting
        weighted_required_score = 0
        weighted_preferred_score = 0
        
        for skill in required_matches:
            weight = self._get_skill_weight(skill, resume_skill_categories)
            weighted_required_score += weight
        
        for skill in preferred_matches:
            weight = self._get_skill_weight(skill, resume_skill_categories)
            weighted_preferred_score += weight
        
        # Calculate final scores
        required_total_weight = sum(self._get_skill_weight(skill, resume_skill_categories) 
                                  for skill in jd_required_skills)
        preferred_total_weight = sum(self._get_skill_weight(skill, resume_skill_categories) 
                                   for skill in jd_preferred_skills)
        
        required_score = (weighted_required_score / required_total_weight * 100) if required_total_weight > 0 else 100
        preferred_score = (weighted_preferred_score / preferred_total_weight * 100) if preferred_total_weight > 0 else 100
        
        # Weighted combination (required skills more important)
        if jd_required_skills and jd_preferred_skills:
            final_score = required_score * 0.75 + preferred_score * 0.25
        elif jd_required_skills:
            final_score = required_score
        else:
            final_score = preferred_score
        
        return min(final_score, 100.0)
    
    def _get_skill_weight(self, skill: str, resume_skill_categories: Dict) -> float:
        """Get weight for a skill based on its category and importance"""
        base_weight = 1.0
        
        # Find skill category and apply weight
        for category, skills in resume_skill_categories.items():
            if skill in skills:
                category_weight = self.skill_importance.get(category, 1.0)
                return base_weight * category_weight
        
        return base_weight
    
    def _calculate_semantic_similarity(self, resume_text: str, jd_text: str) -> float:
        """Enhanced semantic similarity calculation"""
        try:
            if self.sentence_model and len(resume_text) > 100 and len(jd_text) > 100:
                # Use sentence transformers for better semantic understanding
                resume_embedding = self.sentence_model.encode([resume_text])
                jd_embedding = self.sentence_model.encode([jd_text])
                
                similarity = cosine_similarity(resume_embedding, jd_embedding)[0][0]
                return similarity * 100
            else:
                # Enhanced TF-IDF approach
                return self._calculate_enhanced_tfidf_similarity(resume_text, jd_text)
                
        except Exception as e:
            logging.error(f"Error in semantic similarity: {str(e)}")
            return self._calculate_enhanced_tfidf_similarity(resume_text, jd_text)
    
    def _calculate_enhanced_tfidf_similarity(self, resume_text: str, jd_text: str) -> float:
        """Enhanced TF-IDF based similarity with preprocessing"""
        try:
            # Preprocess texts
            resume_processed = self._preprocess_text_for_similarity(resume_text)
            jd_processed = self._preprocess_text_for_similarity(jd_text)
            
            documents = [resume_processed, jd_processed]
            tfidf_matrix = self.tfidf_vectorizer.fit_transform(documents)
            similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
            
            # Boost similarity if both texts are substantial
            if len(resume_processed) > 500 and len(jd_processed) > 500:
                similarity *= 1.1
            
            return min(similarity * 100, 100.0)
        except Exception:
            return 50.0  # Default score if calculation fails
    
    def _preprocess_text_for_similarity(self, text: str) -> str:
        """Preprocess text for better similarity calculation"""
        # Convert to lowercase
        text = text.lower()
        
        # Remove common non-informative patterns
        text = re.sub(r'\b(?:experience|years?|skilled?|knowledge|familiar|proficient)\b', '', text)
        
        # Normalize technology terms
        tech_normalizations = {
            'javascript': 'js',
            'typescript': 'ts',
            'python': 'py',
            'artificial intelligence': 'ai',
            'machine learning': 'ml',
            'natural language processing': 'nlp',
            'user interface': 'ui',
            'user experience': 'ux'
        }
        
        for full_term, short_term in tech_normalizations.items():
            text = re.sub(r'\b' + re.escape(full_term) + r'\b', short_term, text)
        
        return text
    
    def _calculate_experience_match(self, parsed_resume: Dict, parsed_jd: Dict) -> float:
        """Enhanced experience matching with domain consideration"""
        jd_experience = parsed_jd.get('experience_required', {})
        min_years_required = jd_experience.get('min_years', 0)
        max_years_required = jd_experience.get('max_years', None)
        specific_experience = jd_experience.get('specific_experience', [])
        
        # Calculate resume experience
        resume_experience = self._estimate_resume_experience(parsed_resume)
        
        if min_years_required == 0 and not specific_experience:
            return 100.0  # No specific experience required
        
        score = 0
        
        # Years-based scoring
        if min_years_required > 0:
            if resume_experience >= min_years_required:
                years_score = 80
                # Bonus for meeting requirements exactly
                if max_years_required and resume_experience <= max_years_required:
                    years_score = 100
                # Moderate bonus for exceeding requirements
                elif resume_experience > min_years_required:
                    excess = min(resume_experience - min_years_required, 5)
                    years_score += excess * 2
            else:
                # Penalty for not meeting requirements
                shortage = min_years_required - resume_experience
                years_score = max(20, 80 - (shortage * 15))
            
            score = years_score * 0.7
        
        # Domain-specific experience scoring
        if specific_experience:
            domain_score = self._calculate_domain_experience_match(parsed_resume, specific_experience)
            score += domain_score * 0.3
        
        # If only domain experience is specified
        if min_years_required == 0 and specific_experience:
            score = self._calculate_domain_experience_match(parsed_resume, specific_experience)
        
        return min(100.0, score)
    
    def _calculate_domain_experience_match(self, parsed_resume: Dict, specific_experience: List[str]) -> float:
        """Calculate domain-specific experience match"""
        resume_text = ' '.join([
            str(exp.get('responsibilities', [])) for exp in parsed_resume.get('experience', [])
        ]).lower()
        
        if not resume_text:
            return 30.0  # Some credit for having a resume
        
        matches = 0
        total_requirements = len(specific_experience)
        
        for exp_req in specific_experience:
            exp_req_lower = exp_req.lower()
            # Check for partial matches
            if any(word in resume_text for word in exp_req_lower.split() if len(word) > 3):
                matches += 1
        
        if total_requirements == 0:
            return 80.0
        
        match_ratio = matches / total_requirements
        return match_ratio * 100
    
    def _estimate_resume_experience(self, parsed_resume: Dict) -> int:
        """Enhanced experience estimation"""
        max_years = 0
        
        # Check direct experience mention
        experience_data = parsed_resume.get('experience', [])
        for exp in experience_data:
            if exp.get('type') == 'total_experience':
                max_years = max(max_years, exp.get('years', 0))
            
            # Parse duration from job entries
            duration = exp.get('duration', '')
            if duration:
                years = self._parse_duration_to_years(duration)
                max_years = max(max_years, years)
        
        # If no direct experience found, estimate from skills and education
        if max_years == 0:
            skill_count = sum(len(skills) for skills in parsed_resume.get('skills', {}).values())
            project_count = len(parsed_resume.get('projects', []))
            cert_count = len(parsed_resume.get('certifications', []))
            
            # Sophisticated estimation based on profile strength
            if skill_count > 25 and project_count > 3:
                max_years = 4
            elif skill_count > 15 and project_count > 2:
                max_years = 2
            elif skill_count > 8 or project_count > 1:
                max_years = 1
            elif cert_count > 2:
                max_years = 1
        
        return max_years
    
    def _parse_duration_to_years(self, duration: str) -> int:
        """Parse duration string to years"""
        duration_lower = duration.lower()
        
        # Extract years directly mentioned
        year_matches = re.findall(r'(\d+)\s*years?', duration_lower)
        if year_matches:
            return max(int(year) for year in year_matches)
        
        # Parse date ranges
        range_matches = re.findall(r'(\d{4})\s*[-–]\s*(\d{4}|present)', duration_lower)
        if range_matches:
            max_duration = 0
            for start, end in range_matches:
                end_year = 2024 if end == 'present' else int(end)
                years = end_year - int(start)
                max_duration = max(max_duration, years)
            return max_duration
        
        # Parse month ranges and convert to years
        month_matches = re.findall(r'(\d+)\s*months?', duration_lower)
        if month_matches:
            max_months = max(int(month) for month in month_matches)
            return max_months // 12
        
        return 0
    
    def _calculate_education_match(self, parsed_resume: Dict, parsed_jd: Dict) -> float:
        """Enhanced education matching"""
        jd_qualifications = [qual.lower() for qual in parsed_jd.get('qualifications', [])]
        resume_education = parsed_resume.get('education', [])
        
        if not jd_qualifications:
            return 100.0  # No specific education required
        
        if not resume_education:
            return 30.0  # Some penalty for missing education info
        
        # Analyze degree levels and fields
        jd_degree_info = self._analyze_degree_requirements(jd_qualifications)
        resume_degree_info = self._analyze_resume_degrees(resume_education)
        
        # Calculate degree level match
        level_score = self._calculate_degree_level_match(jd_degree_info, resume_degree_info)
        
        # Calculate field match
        field_score = self._calculate_field_match(jd_degree_info, resume_degree_info)
        
        # Combine scores
        final_score = level_score * 0.6 + field_score * 0.4
        
        return min(100.0, final_score)
    
    def _analyze_degree_requirements(self, qualifications: List[str]) -> Dict:
        """Analyze degree requirements from JD"""
        degree_info = {
            'required_level': None,
            'preferred_level': None,
            'fields': [],
            'has_experience_alternative': False
        }
        
        degree_levels = {
            'bachelor': ['bachelor', 'b.tech', 'b.sc', 'b.com', 'bba', 'bca', 'be', 'graduation'],
            'master': ['master', 'm.tech', 'm.sc', 'm.com', 'mba', 'mca', 'me', 'post graduation'],
            'phd': ['phd', 'doctorate', 'doctoral']
        }
        
        for qual in qualifications:
            qual_lower = qual.lower()
            
            # Check for degree levels
            for level, variants in degree_levels.items():
                if any(variant in qual_lower for variant in variants):
                    if not degree_info['required_level']:
                        degree_info['required_level'] = level
                    break
            
            # Extract fields
            field_indicators = ['in', 'of', 'from']
            for indicator in field_indicators:
                if indicator in qual_lower:
                    parts = qual_lower.split(indicator)
                    if len(parts) > 1:
                        field = parts[1].strip()
                        if len(field) > 3:
                            degree_info['fields'].append(field)
        
        return degree_info
    
    def _analyze_resume_degrees(self, education: List[Dict]) -> Dict:
        """Analyze degrees from resume"""
        resume_degree_info = {
            'highest_level': None,
            'fields': [],
            'degrees': []
        }
        
        degree_hierarchy = ['phd', 'master', 'bachelor']
        
        for edu in education:
            degree = edu.get('degree', '').lower()
            field = edu.get('field', '').lower()
            
            # Determine degree level
            for level in degree_hierarchy:
                if level in degree or any(variant in degree for variant in {
                    'bachelor': ['bachelor', 'b.tech', 'b.sc', 'b.com', 'bba', 'bca', 'be'],
                    'master': ['master', 'm.tech', 'm.sc', 'm.com', 'mba', 'mca', 'me'],
                    'phd': ['phd', 'doctorate']
                }.get(level, [])):
                    if not resume_degree_info['highest_level']:
                        resume_degree_info['highest_level'] = level
                    break
            
            if field:
                resume_degree_info['fields'].append(field)
            
            resume_degree_info['degrees'].append({
                'degree': degree,
                'field': field
            })
        
        return resume_degree_info
    
    def _calculate_degree_level_match(self, jd_info: Dict, resume_info: Dict) -> float:
        """Calculate degree level matching score"""
        required_level = jd_info.get('required_level')
        highest_level = resume_info.get('highest_level')
        
        if not required_level:
            return 90.0  # No specific level required
        
        if not highest_level:
            return 40.0  # No degree information
        
        level_hierarchy = {'bachelor': 1, 'master': 2, 'phd': 3}
        
        required_rank = level_hierarchy.get(required_level, 1)
        resume_rank = level_hierarchy.get(highest_level, 1)
        
        if resume_rank >= required_rank:
            # Bonus for higher degree
            if resume_rank > required_rank:
                return 100.0
            else:
                return 95.0
        else:
            # Penalty for lower degree
            penalty = (required_rank - resume_rank) * 20
            return max(50.0, 95.0 - penalty)
    
    def _calculate_field_match(self, jd_info: Dict, resume_info: Dict) -> float:
        """Calculate field of study matching score"""
        jd_fields = jd_info.get('fields', [])
        resume_fields = resume_info.get('fields', [])
        
        if not jd_fields:
            return 90.0  # No specific field required
        
        if not resume_fields:
            return 60.0  # No field information
        
        # Check for field matches
        matches = 0
        for jd_field in jd_fields:
            for resume_field in resume_fields:
                if self._fields_match(jd_field, resume_field):
                    matches += 1
                    break
        
        if matches > 0:
            match_ratio = matches / len(jd_fields)
            return 60 + (match_ratio * 40)  # 60-100 range
        
        return 60.0  # No direct field match
    
    def _fields_match(self, jd_field: str, resume_field: str) -> bool:
        """Check if two fields of study match"""
        # Related fields mapping
        related_fields = {
            'computer science': ['information technology', 'software engineering', 'computer engineering'],
            'data science': ['statistics', 'mathematics', 'computer science', 'information technology'],
            'engineering': ['computer science', 'information technology', 'electronics', 'mechanical'],
            'business': ['management', 'commerce', 'economics', 'finance', 'marketing'],
            'mathematics': ['statistics', 'data science', 'computer science', 'physics']
        }
        
        # Direct match
        if jd_field in resume_field or resume_field in jd_field:
            return True
        
        # Check related fields
        for main_field, related in related_fields.items():
            if main_field in jd_field and any(rel in resume_field for rel in related):
                return True
            if main_field in resume_field and any(rel in jd_field for rel in related):
                return True
        
        return False
    
    def _calculate_context_match(self, parsed_resume: Dict, parsed_jd: Dict) -> float:
        """Calculate contextual and domain relevance match"""
        context_score = 0
        
        # Industry/domain match based on project descriptions
        resume_projects = parsed_resume.get('projects', [])
        jd_responsibilities = parsed_jd.get('responsibilities', [])
        
        if resume_projects and jd_responsibilities:
            domain_score = self._calculate_domain_overlap(resume_projects, jd_responsibilities)
            context_score += domain_score * 0.4
        
        # Technology stack alignment
        tech_score = self._calculate_technology_stack_alignment(parsed_resume, parsed_jd)
        context_score += tech_score * 0.3
        
        # Career progression relevance
        progression_score = self._calculate_career_progression_relevance(parsed_resume, parsed_jd)
        context_score += progression_score * 0.3
        
        return min(100.0, context_score)
    
    def _calculate_domain_overlap(self, projects: List[Dict], responsibilities: List[str]) -> float:
        """Calculate domain overlap between projects and job responsibilities"""
        if not projects or not responsibilities:
            return 50.0
        
        project_text = ' '.join([
            proj.get('description', '') + ' ' + proj.get('title', '')
            for proj in projects
        ]).lower()
        
        responsibility_text = ' '.join(responsibilities).lower()
        
        # Extract domain keywords
        domain_keywords = [
            'web', 'mobile', 'data', 'analytics', 'machine learning', 'ai', 'cloud',
            'backend', 'frontend', 'fullstack', 'devops', 'security', 'testing',
            'api', 'database', 'microservices', 'automation', 'dashboard'
        ]
        
        project_domains = set()
        responsibility_domains = set()
        
        for keyword in domain_keywords:
            if keyword in project_text:
                project_domains.add(keyword)
            if keyword in responsibility_text:
                responsibility_domains.add(keyword)
        
        if not responsibility_domains:
            return 70.0  # No clear domain requirements
        
        overlap = len(project_domains.intersection(responsibility_domains))
        total_domains = len(responsibility_domains)
        
        return (overlap / total_domains) * 100 if total_domains > 0 else 50.0
    
    def _calculate_technology_stack_alignment(self, parsed_resume: Dict, parsed_jd: Dict) -> float:
        """Calculate technology stack alignment"""
        # Get technology skills from both resume and JD
        resume_tech_skills = set()
        jd_tech_skills = set()
        
        tech_categories = ['programming_languages', 'web_technologies', 'databases', 'cloud_platforms', 'devops_tools']
        
        resume_skills = parsed_resume.get('skills', {})
        for category in tech_categories:
            if category in resume_skills:
                resume_tech_skills.update([skill.lower() for skill in resume_skills[category]])
        
        jd_required = [skill.lower() for skill in parsed_jd.get('required_skills', [])]
        jd_preferred = [skill.lower() for skill in parsed_jd.get('preferred_skills', [])]
        jd_tech_skills.update(jd_required + jd_preferred)
        
        if not jd_tech_skills:
            return 70.0
        
        overlap = len(resume_tech_skills.intersection(jd_tech_skills))
        return min(100.0, (overlap / len(jd_tech_skills)) * 120)  # Allow bonus for strong alignment
    
    def _calculate_career_progression_relevance(self, parsed_resume: Dict, parsed_jd: Dict) -> float:
        """Calculate career progression relevance"""
        # This is a simplified implementation
        # In a full implementation, this would analyze career trajectory
        
        resume_experience = self._estimate_resume_experience(parsed_resume)
        jd_experience = parsed_jd.get('experience_required', {}).get('min_years', 0)
        
        if jd_experience == 0:
            return 80.0  # Entry level position
        
        # Check if candidate is overqualified or underqualified
        if resume_experience >= jd_experience * 0.8:
            return 90.0  # Good match
        elif resume_experience >= jd_experience * 0.5:
            return 75.0  # Acceptable with potential
        else:
            return 60.0  # May need significant growth
    
    def _get_detailed_skill_analysis(self, parsed_resume: Dict, parsed_jd: Dict) -> Dict:
        """Get detailed skill analysis with categorization"""
        jd_required = set([skill.lower() for skill in parsed_jd.get('required_skills', [])])
        jd_preferred = set([skill.lower() for skill in parsed_jd.get('preferred_skills', [])])
        jd_all_skills = jd_required.union(jd_preferred)
        
        resume_skills = set()
        resume_skill_details = {}
        
        for category, skills in parsed_resume.get('skills', {}).items():
            for skill in skills:
                skill_lower = skill.lower()
                resume_skills.add(skill_lower)
                resume_skill_details[skill_lower] = {
                    'original': skill,
                    'category': category
                }
        
        # Find matches and gaps
        matching_skills = []
        critical_missing = []
        nice_to_have_missing = []
        
        for skill in jd_all_skills:
            if skill in resume_skills:
                matching_skills.append({
                    'skill': resume_skill_details[skill]['original'],
                    'category': resume_skill_details[skill]['category'],
                    'importance': 'required' if skill in jd_required else 'preferred'
                })
        
        for skill in jd_required:
            if skill not in resume_skills:
                critical_missing.append(skill)
        
        for skill in jd_preferred:
            if skill not in resume_skills:
                nice_to_have_missing.append(skill)
        
        # Identify strengths (resume skills not in JD but valuable)
        additional_strengths = []
        valuable_skills = {'leadership', 'project management', 'mentoring', 'architecture', 'system design'}
        
        for skill in resume_skills:
            if skill not in jd_all_skills and skill in valuable_skills:
                additional_strengths.append(resume_skill_details[skill]['original'])
        
        return {
            'matching_skills': [match['skill'] for match in matching_skills],
            'missing_skills': critical_missing,
            'skill_gaps': {
                'critical': critical_missing,
                'nice_to_have': nice_to_have_missing
            },
            'strengths': additional_strengths,
            'detailed_matches': matching_skills
        }
    
    def _get_enhanced_verdict(self, score: float, matching_analysis: Dict) -> str:
        """Enhanced verdict determination with additional factors"""
        critical_missing = len(matching_analysis['skill_gaps']['critical'])
        matching_count = len(matching_analysis['matching_skills'])
        
        # Adjust verdict based on critical missing skills
        if score >= 75 and critical_missing <= 1:
            return "High"
        elif score >= 60 and critical_missing <= 3:
            return "Medium"
        elif score >= 45 and matching_count >= 3:
            return "Medium"
        else:
            return "Low"
    
    def _generate_personalized_suggestions(self, parsed_resume: Dict, parsed_jd: Dict, 
                                         score: float, matching_analysis: Dict) -> List[str]:
        """Generate personalized improvement suggestions"""
        suggestions = []
        
        critical_missing = matching_analysis['skill_gaps']['critical']
        nice_to_have_missing = matching_analysis['skill_gaps']['nice_to_have']
        
        # Skill-based suggestions with priority
        if critical_missing:
            top_critical = critical_missing[:3]
            suggestions.append(f"🎯 **Priority Skills**: Master these critical skills: {', '.join(top_critical)}")
            
            # Specific learning suggestions for popular technologies
            tech_learning_paths = {
                'python': "Start with Python basics on Codecademy or freeCodeCamp, then build projects",
                'react': "Learn React through the official tutorial, then build a portfolio project",
                'aws': "Get AWS Cloud Practitioner certification, then focus on EC2 and S3",
                'machine learning': "Complete Andrew Ng's ML course on Coursera, practice on Kaggle",
                'docker': "Follow Docker's official tutorial, containerize a simple application"
            }
            
            for skill in top_critical[:2]:
                if skill in tech_learning_paths:
                    suggestions.append(f"📚 **{skill.title()} Learning Path**: {tech_learning_paths[skill]}")
        
        # Experience-based suggestions
        resume_exp = self._estimate_resume_experience(parsed_resume)
        required_exp = parsed_jd.get('experience_required', {}).get('min_years', 0)
        
        if resume_exp < required_exp:
            years_short = required_exp - resume_exp
            if years_short <= 2:
                suggestions.append(f"💼 **Experience Gap**: Consider freelance projects or open-source contributions to bridge the {years_short}-year experience gap")
            else:
                suggestions.append(f"💼 **Experience Building**: This role requires {years_short} more years of experience. Consider similar but junior positions first")
        
        # Project-based suggestions
        resume_projects = len(parsed_resume.get('projects', []))
        if resume_projects < 3:
            suggestions.append("🚀 **Portfolio Projects**: Build 2-3 substantial projects showcasing the required skills")
            
            if critical_missing:
                suggestions.append(f"💡 **Project Ideas**: Create projects using {', '.join(critical_missing[:2])} to demonstrate practical skills")
        
        # Certification suggestions
        if score < 60:
            cert_suggestions = {
                'aws': "AWS Certified Cloud Practitioner",
                'azure': "Microsoft Azure Fundamentals",
                'python': "Python Institute PCAP Certification",
                'project management': "Google Project Management Certificate",
                'data science': "IBM Data Science Professional Certificate"
            }
            
            relevant_certs = []
            for skill in critical_missing[:3]:
                if skill in cert_suggestions:
                    relevant_certs.append(cert_suggestions[skill])
            
            if relevant_certs:
                suggestions.append(f"🏆 **Recommended Certifications**: {', '.join(relevant_certs[:2])}")
        
        # Resume optimization suggestions
        if score < 80:
            suggestions.append("📝 **Resume Optimization**: Add quantified achievements and use keywords from the job description")
            
        if len(matching_analysis['matching_skills']) < 5:
            suggestions.append("🔍 **Skill Highlighting**: Better showcase your transferable skills and relevant experience")
        
        # Career progression suggestions
        if score >= 70:
            suggestions.append("🎯 **You're a strong candidate!** Consider applying and highlighting your matching skills in the cover letter")
        elif score >= 50:
            suggestions.append("📈 **Growth Opportunity**: With focused upskilling in key areas, you could be an excellent fit")
        
        return suggestions[:8]  # Return top 8 suggestions
    
    def _calculate_confidence_score(self, parsed_resume: Dict, parsed_jd: Dict, relevance_score: float) -> float:
        """Calculate confidence in the relevance analysis"""
        confidence_factors = []
        
        # Resume completeness
        resume_sections = [
            'contact_info', 'skills', 'experience', 'education', 'projects'
        ]
        complete_sections = sum(1 for section in resume_sections 
                              if parsed_resume.get(section) and len(str(parsed_resume[section])) > 10)
        
        resume_completeness = (complete_sections / len(resume_sections)) * 100
        confidence_factors.append(resume_completeness * 0.3)
        
        # JD clarity
        jd_sections = ['required_skills', 'responsibilities', 'experience_required']
        clear_sections = sum(1 for section in jd_sections 
                           if parsed_jd.get(section) and len(str(parsed_jd[section])) > 5)
        
        jd_clarity = (clear_sections / len(jd_sections)) * 100
        confidence_factors.append(jd_clarity * 0.3)
        
        # Skill overlap quality
        skill_overlap = len(self._get_detailed_skill_analysis(parsed_resume, parsed_jd)['matching_skills'])
        skill_confidence = min(100, skill_overlap * 10)
        confidence_factors.append(skill_confidence * 0.4)
        
        return sum(confidence_factors)
    
    def _generate_comprehensive_summary(self, score: float, matching_analysis: Dict, 
                                      confidence_score: float) -> str:
        """Generate comprehensive analysis summary"""
        verdict = self._get_enhanced_verdict(score, matching_analysis)
        matching_count = len(matching_analysis['matching_skills'])
        critical_missing = len(matching_analysis['skill_gaps']['critical'])
        
        summary = f"**Overall Assessment: {verdict} Suitability ({score:.0f}/100)**\n\n"
        
        if confidence_score >= 80:
            summary += "🔍 **High Confidence Analysis** - Comprehensive resume and job description data available.\n\n"
        elif confidence_score >= 60:
            summary += "🔍 **Medium Confidence Analysis** - Good data quality for reliable assessment.\n\n"
        else:
            summary += "🔍 **Limited Confidence** - Some information gaps may affect accuracy.\n\n"
        
        # Strengths section
        if matching_count > 0:
            summary += f"**✅ Key Strengths:** Strong alignment in {matching_count} skill areas"
            if matching_count >= 5:
                summary += " - excellent technical match"
            elif matching_count >= 3:
                summary += " - good foundation present"
            summary += ".\n\n"
        
        # Areas for improvement
        if critical_missing > 0:
            summary += f"**🎯 Development Areas:** {critical_missing} critical skills need attention"
            if critical_missing >= 5:
                summary += " - significant upskilling required"
            elif critical_missing >= 3:
                summary += " - moderate skill development needed"
            else:
                summary += " - minor gaps to address"
            summary += ".\n\n"
        
        # Recommendation
        if score >= 75:
            summary += "**💪 Recommendation:** Strong candidate - proceed with application and emphasize matching skills."
        elif score >= 60:
            summary += "**📈 Recommendation:** Promising candidate with growth potential - focus on key skill gaps."
        elif score >= 45:
            summary += "**🎯 Recommendation:** Potential fit with targeted upskilling in critical areas."
        else:
            summary += "**📚 Recommendation:** Significant preparation needed - consider this as a future goal."
        
        return summary
    
    def _generate_improvement_roadmap(self, matching_analysis: Dict) -> Dict:
        """Generate structured improvement roadmap"""
        roadmap = {
            'immediate': [],
            'short_term': [],
            'long_term': []
        }
        
        critical_missing = matching_analysis['skill_gaps']['critical']
        nice_to_have = matching_analysis['skill_gaps']['nice_to_have']
        
        # Immediate actions (1-2 weeks)
        if critical_missing:
            roadmap['immediate'].append("Research and understand the critical missing skills")
            roadmap['immediate'].append("Set up learning accounts on relevant platforms")
        
        # Short-term goals (1-3 months)
        if critical_missing:
            top_skills = critical_missing[:3]
            roadmap['short_term'].append(f"Complete beginner courses in: {', '.join(top_skills)}")
            roadmap['short_term'].append("Build small practice projects using new skills")
        
        # Long-term goals (3-6 months)
        if len(critical_missing) > 3:
            remaining_skills = critical_missing[3:]
            roadmap['long_term'].append(f"Master additional skills: {', '.join(remaining_skills)}")
        
        if nice_to_have:
            roadmap['long_term'].append(f"Explore preferred skills: {', '.join(nice_to_have[:3])}")
        
        roadmap['long_term'].append("Build a comprehensive portfolio showcasing all skills")
        roadmap['long_term'].append("Apply for the target role or similar positions")
        
        return roadmap
    
    def _generate_market_insights(self, parsed_jd: Dict, score: float) -> Dict:
        """Generate market insights and competition analysis"""
        insights = {
            'competition_level': 'Medium',
            'skill_demand': {},
            'advice': []
        }
        
        required_skills = parsed_jd.get('required_skills', [])
        experience_req = parsed_jd.get('experience_required', {}).get('min_years', 0)
        
        # Analyze competition level
        high_demand_skills = ['python', 'javascript', 'react', 'aws', 'machine learning', 'data science']
        rare_skills = ['rust', 'blockchain', 'quantum computing', 'webassembly']
        
        high_demand_count = sum(1 for skill in required_skills if skill.lower() in high_demand_skills)
        rare_skill_count = sum(1 for skill in required_skills if skill.lower() in rare_skills)
        
        if rare_skill_count > 0 or experience_req >= 7:
            insights['competition_level'] = 'Low'
            insights['advice'].append("Specialized position with limited competition")
        elif high_demand_count >= 3 and experience_req <= 3:
            insights['competition_level'] = 'High'
            insights['advice'].append("Popular tech stack - expect high competition")
        
        # Skill demand analysis
        for skill in required_skills:
            if skill.lower() in high_demand_skills:
                insights['skill_demand'][skill] = 'High'
            elif skill.lower() in rare_skills:
                insights['skill_demand'][skill] = 'Rare'
            else:
                insights['skill_demand'][skill] = 'Medium'
        
        # Market advice
        if score >= 70:
            insights['advice'].append("Strong profile for current market demand")
        elif score >= 50:
            insights['advice'].append("Competitive with focused skill development")
        else:
            insights['advice'].append("Consider building skills in high-demand areas first")
        
        return insights
    
    def _get_default_analysis(self) -> Dict:
        """Return default analysis in case of errors"""
        return {
            'relevance_score': 50,
            'verdict': "Medium",
            'confidence_score': 30,
            'detailed_scores': {
                'hard_match': 50,
                'semantic_match': 50,
                'experience_match': 50,
                'education_match': 50,
                'context_match': 50
            },
            'matching_skills': [],
            'missing_skills': [],
            'skill_gaps': {'critical': [], 'nice_to_have': []},
            'strengths': [],
            'suggestions': ["Analysis could not be completed due to technical issues. Please try again."],
            'analysis_summary': "Analysis could not be completed due to technical issues.",
            'improvement_roadmap': {'immediate': [], 'short_term': [], 'long_term': []},
            'market_insights': {'competition_level': 'Medium', 'skill_demand': {}, 'advice': []}
        }
    
    def batch_analyze_resumes(self, resumes_data: List[Dict], jd_data: Dict) -> List[Dict]:
        """Analyze multiple resumes against a single job description"""
        results = []
        
        for i, resume_data in enumerate(resumes_data):
            try:
                st.progress((i + 1) / len(resumes_data), text=f"Analyzing resume {i + 1} of {len(resumes_data)}")
                
                analysis = self.analyze_relevance(
                    resume_data['parsed_resume'],
                    jd_data['parsed_data'],
                    resume_data['resume_text'],
                    jd_data['raw_description']
                )
                
                results.append({
                    'candidate_info': resume_data['candidate_info'],
                    'analysis': analysis
                })
                
            except Exception as e:
                logging.error(f"Error analyzing resume for {resume_data.get('candidate_info', {}).get('name', 'Unknown')}: {str(e)}")
                results.append({
                    'candidate_info': resume_data['candidate_info'],
                    'analysis': self._get_default_analysis()
                })
        
        # Sort by relevance score (descending)
        results.sort(key=lambda x: x['analysis']['relevance_score'], reverse=True)
        
        return results
    
    def get_top_candidates(self, analysis_results: List[Dict], top_n: int = 10) -> List[Dict]:
        """Get top N candidates based on comprehensive scoring"""
        # Enhanced sorting that considers multiple factors
        def comprehensive_score(result):
            analysis = result['analysis']
            base_score = analysis['relevance_score']
            confidence = analysis['confidence_score']
            
            # Boost score based on confidence
            adjusted_score = base_score * (confidence / 100)
            
            # Additional factors
            matching_skills_count = len(analysis.get('matching_skills', []))
            critical_missing_count = len(analysis.get('skill_gaps', {}).get('critical', []))
            
            # Bonus for many matching skills
            skill_bonus = min(matching_skills_count * 2, 20)
            
            # Penalty for critical missing skills
            skill_penalty = critical_missing_count * 5
            
            final_score = adjusted_score + skill_bonus - skill_penalty
            
            return final_score
        
        sorted_results = sorted(analysis_results, key=comprehensive_score, reverse=True)
        return sorted_results[:top_n]
    
    def generate_hiring_insights(self, analysis_results: List[Dict], jd_data: Dict) -> Dict:
        """Generate hiring insights for HR team"""
        if not analysis_results:
            return {
                'total_candidates': 0,
                'quality_distribution': {},
                'skill_gaps_analysis': {},
                'recommendations': []
            }
        
        insights = {
            'total_candidates': len(analysis_results),
            'quality_distribution': {},
            'skill_gaps_analysis': {},
            'common_strengths': [],
            'common_gaps': [],
            'recommendations': [],
            'salary_suggestions': {},
            'timeline_estimates': {}
        }
        
        # Analyze quality distribution
        verdict_counts = {'High': 0, 'Medium': 0, 'Low': 0}
        score_sum = 0
        
        for result in analysis_results:
            verdict = result['analysis']['verdict']
            verdict_counts[verdict] += 1
            score_sum += result['analysis']['relevance_score']
        
        insights['quality_distribution'] = verdict_counts
        insights['average_score'] = score_sum / len(analysis_results)
        
        # Analyze common skill gaps
        all_missing_skills = []
        all_matching_skills = []
        
        for result in analysis_results:
            missing = result['analysis'].get('skill_gaps', {}).get('critical', [])
            matching = result['analysis'].get('matching_skills', [])
            all_missing_skills.extend(missing)
            all_matching_skills.extend(matching)
        
        # Find most common gaps and strengths
        missing_counter = Counter(all_missing_skills)
        matching_counter = Counter(all_matching_skills)
        
        insights['common_gaps'] = missing_counter.most_common(5)
        insights['common_strengths'] = matching_counter.most_common(5)
        
        # Generate recommendations
        high_quality_ratio = verdict_counts['High'] / len(analysis_results)
        
        if high_quality_ratio >= 0.3:
            insights['recommendations'].append("Excellent candidate pool - proceed with interviews for top candidates")
        elif high_quality_ratio >= 0.1:
            insights['recommendations'].append("Good candidate quality - consider top performers and those with growth potential")
        else:
            insights['recommendations'].append("Limited high-quality matches - consider revising requirements or extending search")
        
        # Skill gap recommendations
        if insights['common_gaps']:
            top_gap = insights['common_gaps'][0][0]
            insights['recommendations'].append(f"Consider training programs for {top_gap} - common gap across candidates")
        
        # Timeline estimates
        if verdict_counts['High'] >= 3:
            insights['timeline_estimates']['time_to_hire'] = "2-3 weeks"
        elif verdict_counts['High'] + verdict_counts['Medium'] >= 5:
            insights['timeline_estimates']['time_to_hire'] = "4-6 weeks"
        else:
            insights['timeline_estimates']['time_to_hire'] = "6+ weeks"
        
        return insights
    
    def compare_candidates(self, candidate1: Dict, candidate2: Dict) -> Dict:
        """Compare two candidates side by side"""
        comparison = {
            'candidate1': candidate1['candidate_info'],
            'candidate2': candidate2['candidate_info'],
            'scores': {
                'candidate1': candidate1['analysis']['relevance_score'],
                'candidate2': candidate2['analysis']['relevance_score']
            },
            'strengths_comparison': {},
            'weaknesses_comparison': {},
            'recommendation': ''
        }
        
        # Compare skills
        skills1 = set(candidate1['analysis'].get('matching_skills', []))
        skills2 = set(candidate2['analysis'].get('matching_skills', []))
        
        comparison['strengths_comparison'] = {
            'candidate1_unique': list(skills1 - skills2),
            'candidate2_unique': list(skills2 - skills1),
            'common_strengths': list(skills1.intersection(skills2))
        }
        
        # Compare gaps
        gaps1 = set(candidate1['analysis'].get('skill_gaps', {}).get('critical', []))
        gaps2 = set(candidate2['analysis'].get('skill_gaps', {}).get('critical', []))
        
        comparison['weaknesses_comparison'] = {
            'candidate1_gaps': list(gaps1),
            'candidate2_gaps': list(gaps2),
            'common_gaps': list(gaps1.intersection(gaps2))
        }
        
        # Generate recommendation
        score1 = candidate1['analysis']['relevance_score']
        score2 = candidate2['analysis']['relevance_score']
        
        if abs(score1 - score2) < 10:
            comparison['recommendation'] = "Both candidates are closely matched - consider interviewing both"
        elif score1 > score2:
            comparison['recommendation'] = f"Candidate 1 is stronger ({score1} vs {score2})"
        else:
            comparison['recommendation'] = f"Candidate 2 is stronger ({score2} vs {score1})"
        
        return comparison
    
    def generate_interview_questions(self, analysis_result: Dict, jd_data: Dict) -> Dict:
        """Generate tailored interview questions based on analysis"""
        questions = {
            'technical_questions': [],
            'experience_questions': [],
            'gap_assessment_questions': [],
            'behavioral_questions': []
        }
        
        matching_skills = analysis_result.get('matching_skills', [])
        missing_skills = analysis_result.get('skill_gaps', {}).get('critical', [])
        
        # Technical questions for matching skills
        for skill in matching_skills[:3]:
            if skill.lower() in ['python', 'javascript', 'java']:
                questions['technical_questions'].append(f"Can you walk me through a recent project where you used {skill}?")
            elif skill.lower() in ['aws', 'azure', 'cloud']:
                questions['technical_questions'].append(f"Describe your experience with {skill} and which services you've used")
            else:
                questions['technical_questions'].append(f"How have you applied {skill} in your previous roles?")
        
        # Gap assessment questions
        for skill in missing_skills[:2]:
            questions['gap_assessment_questions'].append(f"How would you approach learning {skill} if hired?")
            questions['gap_assessment_questions'].append(f"Have you had any exposure to {skill}, even if limited?")
        
        # Experience-based questions
        resume_exp = analysis_result.get('detailed_scores', {}).get('experience_match', 0)
        if resume_exp < 70:
            questions['experience_questions'].append("How do you plan to bridge the experience gap for this role?")
            questions['experience_questions'].append("Describe a situation where you quickly learned new skills")
        
        # Behavioral questions
        questions['behavioral_questions'].extend([
            "Tell me about a challenging technical problem you solved",
            "How do you stay updated with new technologies?",
            "Describe a time when you had to learn something completely new"
        ])
        
        return questions