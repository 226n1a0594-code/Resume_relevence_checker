import re
import spacy
from typing import Dict, List, Set
import streamlit as st

class JDParser:
    def __init__(self):
        self.skills_database = self._load_comprehensive_skills_database()
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            try:
                self.nlp = spacy.load("en_core_web_md")
            except OSError:
                try:
                    self.nlp = spacy.load("en_core_web_lg")
                except OSError:
                    # Silent fallback - no warning message
                    self.nlp = None
    
    def _load_comprehensive_skills_database(self) -> Dict[str, List[str]]:
        """Load comprehensive skills database matching resume parser"""
        return {
            'programming_languages': [
                'python', 'java', 'javascript', 'typescript', 'c++', 'c#', 'c', 'php', 'ruby', 
                'go', 'rust', 'swift', 'kotlin', 'scala', 'r', 'matlab', 'perl', 'lua', 'dart',
                'objective-c', 'assembly', 'cobol', 'fortran', 'haskell', 'erlang', 'elixir',
                'clojure', 'f#', 'visual basic', 'shell scripting', 'bash', 'powershell', 'zsh'
            ],
            'web_technologies': [
                'html', 'css', 'sass', 'scss', 'less', 'bootstrap', 'tailwind css', 'bulma',
                'react', 'angular', 'vue.js', 'svelte', 'ember.js', 'backbone.js', 'jquery',
                'node.js', 'express.js', 'next.js', 'nuxt.js', 'gatsby', 'webpack', 'vite',
                'parcel', 'rollup', 'npm', 'yarn', 'pnpm', 'rest api', 'graphql', 'apollo',
                'websockets', 'ajax', 'json', 'xml', 'soap'
            ],
            'backend_frameworks': [
                'django', 'flask', 'fastapi', 'tornado', 'pyramid', 'spring boot', 'spring mvc',
                'laravel', 'symfony', 'codeigniter', 'ruby on rails', 'sinatra', 'asp.net',
                '.net core', 'express.js', 'koa.js', 'nestjs', 'adonis.js', 'meteor'
            ],
            'databases': [
                'mysql', 'postgresql', 'sqlite', 'oracle', 'sql server', 'mongodb', 'cassandra',
                'redis', 'elasticsearch', 'neo4j', 'couchdb', 'dynamodb', 'firebase firestore',
                'firebase realtime database', 'influxdb', 'clickhouse', 'snowflake', 'bigquery',
                'amazon rds', 'amazon aurora', 'mariadb', 'cockroachdb', 'prisma', 'typeorm',
                'sequelize', 'mongoose', 'sql', 'nosql'
            ],
            'cloud_platforms': [
                'aws', 'amazon web services', 'azure', 'microsoft azure', 'google cloud platform',
                'gcp', 'google cloud', 'alibaba cloud', 'ibm cloud', 'oracle cloud', 'digitalocean',
                'linode', 'vultr', 'heroku', 'netlify', 'vercel', 'cloudflare', 'firebase'
            ],
            'cloud_services': [
                'ec2', 's3', 'rds', 'lambda', 'cloudformation', 'cloudwatch', 'iam', 'vpc',
                'elastic beanstalk', 'ecs', 'eks', 'fargate', 'api gateway', 'cognito',
                'azure functions', 'azure storage', 'azure sql', 'azure cosmos db',
                'compute engine', 'cloud storage', 'cloud functions', 'cloud run',
                'kubernetes engine', 'app engine', 'cloud sql', 'pub/sub'
            ],
            'devops_tools': [
                'docker', 'kubernetes', 'jenkins', 'gitlab ci', 'github actions', 'travis ci',
                'circleci', 'ansible', 'terraform', 'chef', 'puppet', 'vagrant', 'helm',
                'prometheus', 'grafana', 'elk stack', 'elasticsearch', 'logstash', 'kibana',
                'nagios', 'zabbix', 'datadog', 'new relic', 'splunk', 'nginx', 'apache',
                'load balancing', 'reverse proxy', 'microservices', 'serverless', 'istio'
            ],
            'version_control': [
                'git', 'github', 'gitlab', 'bitbucket', 'svn', 'mercurial', 'perforce',
                'git flow', 'github flow', 'branching strategy', 'pull request', 'merge request'
            ],
            'data_science_ml': [
                'machine learning', 'deep learning', 'artificial intelligence', 'neural networks',
                'natural language processing', 'nlp', 'computer vision', 'reinforcement learning',
                'supervised learning', 'unsupervised learning', 'classification', 'regression',
                'clustering', 'dimensionality reduction', 'feature engineering', 'model selection',
                'cross validation', 'hyperparameter tuning', 'ensemble methods', 'random forest',
                'gradient boosting', 'xgboost', 'lightgbm', 'catboost', 'svm', 'support vector machine',
                'decision trees', 'naive bayes', 'k-means', 'hierarchical clustering', 'dbscan',
                'pca', 'principal component analysis', 'linear regression', 'logistic regression'
            ],
            'ml_frameworks': [
                'tensorflow', 'pytorch', 'keras', 'scikit-learn', 'pandas', 'numpy', 'scipy',
                'matplotlib', 'seaborn', 'plotly', 'bokeh', 'opencv', 'pillow', 'scikit-image',
                'nltk', 'spacy', 'gensim', 'transformers', 'hugging face', 'fastai', 'jax',
                'mlflow', 'kubeflow', 'weights and biases', 'wandb', 'tensorboard'
            ],
            'data_tools': [
                'tableau', 'power bi', 'qlikview', 'looker', 'superset', 'metabase', 'grafana',
                'jupyter', 'jupyter notebook', 'jupyter lab', 'google colab', 'kaggle',
                'databricks', 'apache spark', 'pyspark', 'hadoop', 'hive', 'pig', 'hbase',
                'kafka', 'airflow', 'luigi', 'prefect', 'dbt', 'great expectations'
            ],
            'mobile_development': [
                'android', 'ios', 'react native', 'flutter', 'xamarin', 'ionic', 'cordova',
                'phonegap', 'native script', 'unity', 'unreal engine', 'android studio',
                'xcode', 'java android', 'kotlin android', 'swift ios', 'objective-c ios'
            ],
            'testing': [
                'unit testing', 'integration testing', 'end-to-end testing', 'selenium', 'cypress',
                'jest', 'mocha', 'chai', 'jasmine', 'karma', 'protractor', 'playwright',
                'testng', 'junit', 'pytest', 'unittest', 'mockito', 'sinon', 'enzyme',
                'react testing library', 'postman', 'insomnia', 'newman', 'k6', 'locust'
            ],
            'project_management': [
                'agile', 'scrum', 'kanban', 'waterfall', 'lean', 'six sigma', 'prince2',
                'pmp', 'jira', 'confluence', 'trello', 'asana', 'monday.com', 'notion',
                'slack', 'microsoft teams', 'zoom', 'miro', 'figma', 'sketch', 'invision'
            ],
            'security': [
                'cybersecurity', 'information security', 'network security', 'web security',
                'application security', 'penetration testing', 'vulnerability assessment',
                'ethical hacking', 'ceh', 'cissp', 'cism', 'cisa', 'owasp', 'ssl', 'tls',
                'encryption', 'cryptography', 'oauth', 'jwt', 'saml', 'ldap', 'active directory'
            ],
            'design_tools': [
                'adobe photoshop', 'adobe illustrator', 'adobe xd', 'figma', 'sketch',
                'invision', 'principle', 'framer', 'zeplin', 'marvel', 'canva', 'gimp',
                'inkscape', 'blender', 'maya', 'autocad', 'solidworks', 'fusion 360'
            ],
            'soft_skills': [
                'leadership', 'communication', 'teamwork', 'collaboration', 'problem solving',
                'analytical thinking', 'critical thinking', 'creative thinking', 'time management',
                'project management', 'adaptability', 'flexibility', 'mentoring', 'coaching',
                'presentation skills', 'public speaking', 'negotiation', 'conflict resolution',
                'decision making', 'strategic thinking', 'emotional intelligence', 'empathy'
            ],
            'certifications': [
                'aws certified', 'azure certified', 'google cloud certified', 'comptia',
                'cisco certified', 'microsoft certified', 'oracle certified', 'salesforce certified',
                'pmp', 'scrum master', 'product owner', 'itil', 'lean six sigma', 'ceh',
                'cissp', 'cism', 'cisa', 'togaf', 'cobit', 'prince2'
            ],
            'business_skills': [
                'business analysis', 'business intelligence', 'market research', 'competitive analysis',
                'financial modeling', 'budgeting', 'forecasting', 'risk management', 'compliance',
                'stakeholder management', 'vendor management', 'contract negotiation', 'sales',
                'marketing', 'customer relationship management', 'crm', 'erp', 'supply chain'
            ]
        }
    
    def parse_job_description(self, jd_text: str) -> Dict:
        """Enhanced job description parsing with comprehensive analysis"""
        if not jd_text.strip():
            return self._get_empty_jd_structure()
        
        # Clean and preprocess text
        cleaned_text = self._clean_and_normalize_text(jd_text)
        
        parsed_data = {
            'required_skills': self._extract_required_skills(cleaned_text),
            'preferred_skills': self._extract_preferred_skills(cleaned_text),
            'qualifications': self._extract_qualifications(cleaned_text),
            'experience_required': self._extract_experience_requirements(cleaned_text),
            'responsibilities': self._extract_responsibilities(cleaned_text),
            'company_benefits': self._extract_benefits(cleaned_text),
            'job_type_details': self._extract_job_type_details(cleaned_text),
            'keywords': self._extract_all_keywords(cleaned_text),
            'must_have_vs_nice_to_have': self._categorize_requirements(cleaned_text),
            'salary_range': self._extract_salary_range(cleaned_text),
            'location_details': self._extract_location_details(cleaned_text),
            'company_info': self._extract_company_info(cleaned_text),
            'application_deadline': self._extract_application_deadline(cleaned_text),
            'job_summary': self._extract_job_summary(cleaned_text)
        }
        
        # Calculate job complexity score
        parsed_data['complexity_score'] = self._calculate_job_complexity(parsed_data)
        
        return parsed_data
    
    def _get_empty_jd_structure(self) -> Dict:
        """Return empty JD structure"""
        return {
            'required_skills': [],
            'preferred_skills': [],
            'qualifications': [],
            'experience_required': {'min_years': 0, 'max_years': None, 'specific_experience': []},
            'responsibilities': [],
            'company_benefits': [],
            'job_type_details': {},
            'keywords': [],
            'must_have_vs_nice_to_have': {'must_have': [], 'nice_to_have': []},
            'salary_range': {},
            'location_details': {},
            'company_info': {},
            'application_deadline': '',
            'job_summary': '',
            'complexity_score': 0
        }
    
    def _clean_and_normalize_text(self, text: str) -> str:
        """Clean and normalize job description text"""
        if not text:
            return ""
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\n+', '\n', text)
        
        # Normalize common terms
        text = re.sub(r'\b(?:yrs?|years?)\b', 'years', text, flags=re.IGNORECASE)
        text = re.sub(r'\b(?:exp|experience)\b', 'experience', text, flags=re.IGNORECASE)
        text = re.sub(r'\b(?:req|required|mandatory)\b', 'required', text, flags=re.IGNORECASE)
        text = re.sub(r'\b(?:pref|preferred|desirable)\b', 'preferred', text, flags=re.IGNORECASE)
        
        return text.strip()
    
    def _extract_required_skills(self, text: str) -> List[str]:
        """Extract required/must-have skills with enhanced detection"""
        required_skills = set()
        text_lower = text.lower()
        
        # Enhanced patterns for required skills sections
        required_section_patterns = [
            r'(?:required|must\s+have|essential|mandatory|needed|necessary)(?:\s+(?:skills?|qualifications?|requirements?))?\s*:?\s*(.*?)(?=(?:preferred|nice\s+to\s+have|good\s+to\s+have|desirable|optional|bonus|plus|responsibilities|duties|what\s+you.ll\s+do|benefits|salary|location|about|company|apply|send)|$)',
            r'(?:minimum|core|key)\s+(?:requirements?|qualifications?|skills?)\s*:?\s*(.*?)(?=(?:preferred|nice\s+to\s+have|responsibilities|duties|benefits|salary|location|about|company|apply)|$)',
            r'(?:what\s+we.re\s+looking\s+for|candidate\s+should\s+have|you\s+should\s+have)\s*:?\s*(.*?)(?=(?:preferred|nice\s+to\s+have|responsibilities|benefits|salary|location|about|company|apply)|$)'
        ]
        
        # Extract from dedicated required sections
        for pattern in required_section_patterns:
            matches = re.findall(pattern, text_lower, re.DOTALL)
            for match in matches:
                skills = self._extract_skills_from_text_section(match)
                required_skills.update(skills)
        
        # Look for skills with explicit required indicators
        required_indicators = [
            r'(\w+(?:\s+\w+)*)\s+(?:is\s+)?(?:required|mandatory|essential|must\s+have|needed)',
            r'(?:required|mandatory|essential|must\s+have|needed)\s*:?\s*(\w+(?:\s+\w+)*)',
            r'(\w+(?:\s+\w+)*)\s*\(required\)',
            r'(\w+(?:\s+\w+)*)\s*-\s*(?:required|mandatory|essential)',
        ]
        
        for pattern in required_indicators:
            matches = re.findall(pattern, text_lower)
            for match in matches:
                skill = match.strip()
                if self._is_valid_skill(skill):
                    required_skills.add(skill)
        
        # Extract from bullet points and lists in early sections
        early_section = text_lower[:len(text_lower)//2]  # First half of JD
        bullet_patterns = [
            r'[•\*\-\+]\s*([^\n]+)',
            r'\d+[\.\)]\s*([^\n]+)',
            r'(?:^|\n)\s*([A-Z][^.\n]+(?:experience|knowledge|skills?|proficiency))',
        ]
        
        for pattern in bullet_patterns:
            matches = re.findall(pattern, early_section, re.MULTILINE)
            for match in matches:
                skills = self._extract_skills_from_text_section(match)
                required_skills.update(skills)
        
        # If no explicit required section found, extract from general text with higher weight for certain contexts
        if not required_skills:
            skill_context_patterns = [
                r'(\w+(?:\s+\w+)*)\s+(?:experience|knowledge|skills?|expertise|proficiency|background)',
                r'(?:experience|knowledge|skills?|expertise|proficiency|background)\s+(?:in|with|of)\s+(\w+(?:\s+\w+)*)',
                r'(?:strong|solid|good|excellent|deep)\s+(?:experience|knowledge|skills?|expertise)\s+(?:in|with|of)\s+(\w+(?:\s+\w+)*)',
            ]
            
            for pattern in skill_context_patterns:
                matches = re.findall(pattern, text_lower)
                for match in matches:
                    skill = match.strip()
                    if self._is_valid_skill(skill):
                        required_skills.add(skill)
        
        return sorted(list(required_skills))
    
    def _extract_preferred_skills(self, text: str) -> List[str]:
        """Extract preferred/nice-to-have skills"""
        preferred_skills = set()
        text_lower = text.lower()
        
        # Patterns for preferred skills sections
        preferred_section_patterns = [
            r'(?:preferred|nice\s+to\s+have|good\s+to\s+have|desirable|optional|bonus|plus|advantageous|would\s+be\s+nice|ideal|additional)(?:\s+(?:skills?|qualifications?|requirements?))?\s*:?\s*(.*?)(?=(?:required|responsibilities|duties|benefits|salary|location|about|company|apply)|$)',
            r'(?:additional|extra|supplementary)\s+(?:skills?|qualifications?|requirements?)\s*:?\s*(.*?)(?=(?:required|responsibilities|duties|benefits|salary|location|about|company|apply)|$)',
            r'(?:it\s+would\s+be\s+great\s+if|we\s+would\s+love\s+if|bonus\s+points)\s*:?\s*(.*?)(?=(?:required|responsibilities|duties|benefits|salary|location|about|company|apply)|$)'
        ]
        
        for pattern in preferred_section_patterns:
            matches = re.findall(pattern, text_lower, re.DOTALL)
            for match in matches:
                skills = self._extract_skills_from_text_section(match)
                preferred_skills.update(skills)
        
        # Look for skills with explicit preferred indicators
        preferred_indicators = [
            r'(\w+(?:\s+\w+)*)\s+(?:is\s+)?(?:preferred|desirable|nice\s+to\s+have|bonus|plus)',
            r'(?:preferred|desirable|nice\s+to\s+have|bonus|plus)\s*:?\s*(\w+(?:\s+\w+)*)',
            r'(\w+(?:\s+\w+)*)\s*\((?:preferred|optional|bonus)\)',
            r'(\w+(?:\s+\w+)*)\s*-\s*(?:preferred|desirable|optional)',
        ]
        
        for pattern in preferred_indicators:
            matches = re.findall(pattern, text_lower)
            for match in matches:
                skill = match.strip()
                if self._is_valid_skill(skill):
                    preferred_skills.add(skill)
        
        return sorted(list(preferred_skills))
    
    def _extract_skills_from_text_section(self, text_section: str) -> Set[str]:
        """Extract skills from a text section using the skills database"""
        found_skills = set()
        text_lower = text_section.lower()
        
        # Check all skill categories
        for category, skills_list in self.skills_database.items():
            for skill in skills_list:
                # Create flexible pattern for skill matching
                skill_pattern = re.escape(skill.lower()).replace(r'\ ', r'[\s\-_]*')
                pattern = r'\b' + skill_pattern + r'\b'
                
                if re.search(pattern, text_lower):
                    found_skills.add(skill)
        
        # Also extract potential skills using common patterns
        potential_skill_patterns = [
            r'\b([A-Z][a-z]+(?:\.[a-z]+)+)\b',  # Technology names like React.js
            r'\b([A-Z]{2,})\b',  # Acronyms like AWS, API
            r'\b(\w+(?:\+\+|#))\b',  # Programming languages like C++, C#
        ]
        
        for pattern in potential_skill_patterns:
            matches = re.findall(pattern, text_section)
            for match in matches:
                if len(match) > 1 and match.lower() not in ['and', 'the', 'for', 'with', 'from', 'this', 'that']:
                    found_skills.add(match.lower())
        
        return found_skills
    
    def _is_valid_skill(self, skill: str) -> bool:
        """Check if extracted text is a valid skill"""
        skill = skill.strip().lower()
        
        # Length constraints
        if len(skill) < 2 or len(skill) > 50:
            return False
        
        # Check against known skills database
        for category, skills_list in self.skills_database.items():
            if skill in [s.lower() for s in skills_list]:
                return True
        
        # Check if it looks like a technology/skill term
        if re.match(r'^[a-zA-Z][a-zA-Z0-9\+\#\.\-_]*$', skill):
            # Exclude common non-skill words
            excluded_words = [
                'and', 'the', 'for', 'with', 'from', 'this', 'that', 'have', 'been', 'will',
                'can', 'should', 'would', 'could', 'may', 'might', 'must', 'able', 'work',
                'team', 'company', 'role', 'position', 'job', 'candidate', 'person', 'individual'
            ]
            return skill not in excluded_words
        
        return False
    
    def _extract_qualifications(self, text: str) -> List[str]:
        """Extract education and certification requirements"""
        qualifications = []
        text_lower = text.lower()
        
        # Education patterns
        education_patterns = [
            r"bachelor'?s?\s+(?:degree\s+)?(?:in\s+)?([a-zA-Z\s]+)",
            r"master'?s?\s+(?:degree\s+)?(?:in\s+)?([a-zA-Z\s]+)",
            r"(?:phd|doctorate|doctoral)\s+(?:degree\s+)?(?:in\s+)?([a-zA-Z\s]+)",
            r"(?:diploma|certificate)\s+(?:in\s+)?([a-zA-Z\s]+)",
            r"(?:b\.?tech|m\.?tech|mba|bba|bca|mca|b\.?sc|m\.?sc|b\.?com|m\.?com|be|me)\s*(?:in\s+)?([a-zA-Z\s]*)",
            r"(?:graduation|post\s+graduation)\s+(?:in\s+)?([a-zA-Z\s]+)",
            r"(?:undergraduate|graduate)\s+(?:degree\s+)?(?:in\s+)?([a-zA-Z\s]+)"
        ]
        
        for pattern in education_patterns:
            matches = re.findall(pattern, text_lower)
            for match in matches:
                field = match.strip()
                if field and len(field) > 2:
                    qualifications.append(f"Degree in {field.title()}")
        
        # Certification patterns
        cert_patterns = [
            r"(?:certified|certification)\s+([a-zA-Z\s]+)",
            r"([a-zA-Z\s]*certified[a-zA-Z\s]*)",
            r"(aws|azure|google\s+cloud|oracle|microsoft|cisco|comptia|salesforce)\s+(?:certified\s+)?([a-zA-Z\s]+)",
            r"(?:professional\s+)?(?:certification|certificate)\s+(?:in\s+)?([a-zA-Z\s]+)",
        ]
        
        for pattern in cert_patterns:
            matches = re.findall(pattern, text_lower)
            for match in matches:
                if isinstance(match, tuple):
                    cert_text = " ".join(match).strip()
                else:
                    cert_text = match.strip()
                
                if len(cert_text) > 3 and cert_text not in ['and', 'the', 'for', 'with']:
                    qualifications.append(f"Certification: {cert_text.title()}")
        
        # Experience-based qualifications
        exp_qual_patterns = [
            r"(\d+)\+?\s*years?\s+(?:of\s+)?(?:professional\s+)?experience\s+(?:in\s+)?([a-zA-Z\s]+)",
            r"(?:minimum|at\s+least)\s+(\d+)\s*years?\s+(?:experience\s+)?(?:in\s+)?([a-zA-Z\s]+)",
            r"(\d+)\+?\s*years?\s+(?:of\s+)?([a-zA-Z\s]+)\s+experience"
        ]
        
        for pattern in exp_qual_patterns:
            matches = re.findall(pattern, text_lower)
            for match in matches:
                years = match[0]
                field = match[1].strip() if len(match) > 1 else ""
                if field and len(field) > 3:
                    qualifications.append(f"{years}+ years experience in {field.title()}")
        
        return qualifications[:15]  # Top 15 qualifications
    
    def _extract_experience_requirements(self, text: str) -> Dict:
        """Extract detailed experience requirements"""
        experience_req = {
            'min_years': 0,
            'max_years': None,
            'specific_experience': [],
            'level': '',
            'type': ''
        }
        
        text_lower = text.lower()
        
        # Extract years of experience
        year_patterns = [
            r'(\d+)\+?\s*(?:to\s+(\d+))?\s*years?\s+(?:of\s+)?(?:professional\s+)?(?:relevant\s+)?experience',
            r'(\d+)\+?\s*years?\s+(?:of\s+)?(?:professional\s+)?(?:relevant\s+)?experience',
            r'minimum\s+(\d+)\s*years?',
            r'at\s+least\s+(\d+)\s*years?',
            r'(\d+)\s*-\s*(\d+)\s*years?\s+(?:of\s+)?experience',
            r'between\s+(\d+)\s*(?:and|to)\s*(\d+)\s*years?'
        ]
        
        for pattern in year_patterns:
            matches = re.findall(pattern, text_lower)
            for match in matches:
                if isinstance(match, tuple):
                    min_years = int(match[0]) if match[0] else 0
                    max_years = int(match[1]) if len(match) > 1 and match[1] and match[1].isdigit() else None
                else:
                    min_years = int(match) if match.isdigit() else 0
                    max_years = None
                
                experience_req['min_years'] = max(experience_req['min_years'], min_years)
                if max_years:
                    experience_req['max_years'] = max_years
        
        # Extract experience level
        level_patterns = [
            r'\b(entry\s+level|junior|mid\s+level|senior|lead|principal|staff|architect|director|manager)\b',
            r'\b(fresher|experienced|expert)\b'
        ]
        
        for pattern in level_patterns:
            match = re.search(pattern, text_lower)
            if match:
                experience_req['level'] = match.group(1).title()
                break
        
        # Extract specific experience requirements
        exp_indicators = [
            'experience with', 'experience in', 'background in', 'expertise in',
            'familiarity with', 'knowledge of', 'proficiency in', 'hands-on experience',
            'proven experience', 'demonstrated experience', 'working experience'
        ]
        
        for indicator in exp_indicators:
            pattern = f'{re.escape(indicator)}\\s+([^.]+)'
            matches = re.findall(pattern, text_lower)
            for match in matches:
                exp_text = match.strip()
                if len(exp_text) > 5 and len(exp_text) < 100:
                    experience_req['specific_experience'].append(exp_text)
        
        return experience_req
    
    def _extract_responsibilities(self, text: str) -> List[str]:
        """Extract job responsibilities and duties"""
        responsibilities = []
        text_lower = text.lower()
        
        # Responsibility section patterns
        resp_patterns = [
            r'(?:responsibilities|duties|role|what\s+you.ll\s+do|key\s+responsibilities|primary\s+duties|job\s+duties)\s*:?\s*(.*?)(?=(?:requirements|qualifications|skills|experience|benefits|salary|location|about|company|apply|what\s+we\s+offer)|$)',
            r'(?:you\s+will|the\s+role\s+involves|this\s+position|as\s+a\s+[\w\s]+\s+you\s+will)\s*:?\s*(.*?)(?=(?:requirements|qualifications|skills|experience|benefits|salary|location|about|company|apply)|$)',
            r'(?:primary\s+functions|main\s+duties|core\s+responsibilities)\s*:?\s*(.*?)(?=(?:requirements|qualifications|skills|experience|benefits|salary|location|about|company|apply)|$)'
        ]
        
        resp_text = ""
        for pattern in resp_patterns:
            match = re.search(pattern, text_lower, re.DOTALL)
            if match:
                resp_text = match.group(1).strip()
                break
        
        if resp_text:
            # Extract individual responsibilities
            resp_lines = re.split(r'(?:\n|•|\*|\d+\.)', resp_text)
            
            for line in resp_lines:
                line = line.strip()
                if len(line) > 20 and len(line) < 200:
                    # Clean up the responsibility
                    line = re.sub(r'^[-\s]*', '', line)
                    responsibilities.append(line)
        
        return responsibilities[:12]  # Top 12 responsibilities
    
    def _extract_benefits(self, text: str) -> List[str]:
        """Extract company benefits and perks"""
        benefits = []
        text_lower = text.lower()
        
        # Benefit section patterns
        benefit_patterns = [
            r'(?:benefits|perks|what\s+we\s+offer|compensation|package)\s*:?\s*(.*?)(?=(?:requirements|qualifications|responsibilities|about|company|apply|location|salary)|$)',
            r'(?:we\s+offer|our\s+benefits|employee\s+benefits)\s*:?\s*(.*?)(?=(?:requirements|qualifications|responsibilities|about|company|apply)|$)'
        ]
        
        for pattern in benefit_patterns:
            match = re.search(pattern, text_lower, re.DOTALL)
            if match:
                benefit_text = match.group(1).strip()
                benefit_lines = re.split(r'(?:\n|•|\*|\d+\.)', benefit_text)
                
                for line in benefit_lines:
                    line = line.strip()
                    if len(line) > 10 and len(line) < 100:
                        line = re.sub(r'^[-\s]*', '', line)
                        benefits.append(line)
        
        return benefits[:10]  # Top 10 benefits
    
    def _extract_job_type_details(self, text: str) -> Dict:
        """Extract job type and work arrangement details"""
        job_details = {}
        text_lower = text.lower()
        
        # Work arrangement
        if re.search(r'\b(remote|work\s+from\s+home|wfh)\b', text_lower):
            job_details['work_arrangement'] = 'Remote'
        elif re.search(r'\b(hybrid|flexible)\b', text_lower):
            job_details['work_arrangement'] = 'Hybrid'
        elif re.search(r'\b(on-site|office|in-person)\b', text_lower):
            job_details['work_arrangement'] = 'On-site'
        
        # Employment type
        if re.search(r'\b(full.time|full\s+time)\b', text_lower):
            job_details['employment_type'] = 'Full-time'
        elif re.search(r'\b(part.time|part\s+time)\b', text_lower):
            job_details['employment_type'] = 'Part-time'
        elif re.search(r'\b(contract|contractor|freelance)\b', text_lower):
            job_details['employment_type'] = 'Contract'
        elif re.search(r'\b(intern|internship)\b', text_lower):
            job_details['employment_type'] = 'Internship'
        
        # Travel requirements
        travel_match = re.search(r'(\d+)%\s+travel', text_lower)
        if travel_match:
            job_details['travel_required'] = f"{travel_match.group(1)}%"
        elif re.search(r'\b(no\s+travel|minimal\s+travel)\b', text_lower):
            job_details['travel_required'] = 'Minimal'
        elif re.search(r'\b(frequent\s+travel|significant\s+travel)\b', text_lower):
            job_details['travel_required'] = 'Frequent'
        
        return job_details
    
    def _extract_all_keywords(self, text: str) -> List[str]:
        """Extract all relevant keywords from job description"""
        keywords = set()
        text_lower = text.lower()
        
        # Extract all skills as keywords
        for category, skills_list in self.skills_database.items():
            for skill in skills_list:
                skill_pattern = re.escape(skill.lower()).replace(r'\ ', r'[\s\-_]*')
                pattern = r'\b' + skill_pattern + r'\b'
                if re.search(pattern, text_lower):
                    keywords.add(skill)
        
        # Extract technology/company specific keywords
        tech_patterns = [
            r'\b([A-Z][a-zA-Z]*(?:\.[a-zA-Z]+)*)\b',  # Technology names
            r'\b([a-zA-Z]+(?:-[a-zA-Z]+)+)\b',  # Hyphenated terms
            r'\b([A-Z]{2,})\b'  # Acronyms
        ]
        
        for pattern in tech_patterns:
            matches = re.findall(pattern, text)
            for match in matches:
                if len(match) > 2 and match.lower() not in ['and', 'the', 'for', 'with', 'from']:
                    keywords.add(match.lower())
        
        return sorted(list(keywords))
    
    def _categorize_requirements(self, text: str) -> Dict[str, List[str]]:
        """Categorize requirements into must-have vs nice-to-have"""
        categorized = {
            'must_have': [],
            'nice_to_have': []
        }
        
        # Get required and preferred skills
        required_skills = self._extract_required_skills(text)
        preferred_skills = self._extract_preferred_skills(text)
        
        categorized['must_have'] = required_skills
        categorized['nice_to_have'] = preferred_skills
        
        return categorized
    
    def _extract_salary_range(self, text: str) -> Dict:
        """Extract salary information"""
        salary_info = {}
        text_lower = text.lower()
        
        # Salary patterns
        salary_patterns = [
            r'salary\s*:?\s*₹?\s*(\d+(?:,\d+)*)\s*(?:to|-)\s*₹?\s*(\d+(?:,\d+)*)\s*(lpa|per\s+annum|annually)?',
            r'₹\s*(\d+(?:,\d+)*)\s*(?:to|-)\s*₹?\s*(\d+(?:,\d+)*)\s*(lpa|per\s+annum)?',
            r'(\d+)\s*-\s*(\d+)\s*(lpa|lakhs?\s+per\s+annum|k\s+per\s+month)',
            r'compensation\s*:?\s*₹?\s*(\d+(?:,\d+)*)\s*(?:to|-)\s*₹?\s*(\d+(?:,\d+)*)'
        ]
        
        for pattern in salary_patterns:
            match = re.search(pattern, text_lower)
            if match:
                min_salary = match.group(1).replace(',', '')
                max_salary = match.group(2).replace(',', '') if len(match.groups()) > 1 else min_salary
                unit = match.group(3) if len(match.groups()) > 2 else 'lpa'
                
                salary_info['min_salary'] = min_salary
                salary_info['max_salary'] = max_salary
                salary_info['currency'] = 'INR'
                salary_info['unit'] = unit
                break
        
        return salary_info
    
    def _extract_location_details(self, text: str) -> Dict:
        """Extract location and work arrangement details"""
        location_info = {}
        text_lower = text.lower()
        
        # Indian cities pattern
        indian_cities = [
            'mumbai', 'delhi', 'bangalore', 'hyderabad', 'ahmedabad', 'chennai', 'kolkata',
            'surat', 'pune', 'jaipur', 'lucknow', 'kanpur', 'nagpur', 'indore', 'thane',
            'bhopal', 'visakhapatnam', 'pimpri', 'patna', 'vadodara', 'ghaziabad', 'ludhiana',
            'agra', 'nashik', 'faridabad', 'meerut', 'rajkot', 'kalyan', 'vasai', 'varanasi',
            'srinagar', 'dhanbad', 'jodhpur', 'amritsar', 'raipur', 'allahabad', 'coimbatore',
            'jabalpur', 'gwalior', 'vijayawada', 'madurai', 'gurgaon', 'noida', 'navi mumbai'
        ]
        
        # Extract cities
        found_cities = []
        for city in indian_cities:
            if city in text_lower:
                found_cities.append(city.title())
        
        if found_cities:
            location_info['cities'] = found_cities
        
        # Extract specific location patterns
        location_patterns = [
            r'location\s*:?\s*([^\n,]+)',
            r'based\s+(?:in|at)\s+([^\n,]+)',
            r'office\s+(?:in|at|location)\s*:?\s*([^\n,]+)',
            r'work\s+from\s+([^\n,]+)'
        ]
        
        for pattern in location_patterns:
            match = re.search(pattern, text_lower)
            if match:
                location_info['mentioned_location'] = match.group(1).strip().title()
                break
        
        return location_info
    
    def _extract_company_info(self, text: str) -> Dict:
        """Extract company information"""
        company_info = {}
        text_lower = text.lower()
        
        # Company size indicators
        size_patterns = [
            r'(\d+)\+?\s*employees?',
            r'team\s+of\s+(\d+)\+?',
            r'(\d+)\+?\s*people',
            r'(startup|small|medium|large|enterprise)\s+(?:company|organization|firm)'
        ]
        
        for pattern in size_patterns:
            match = re.search(pattern, text_lower)
            if match:
                if match.group(1).isdigit():
                    company_info['size'] = f"{match.group(1)}+ employees"
                else:
                    company_info['size'] = match.group(1).title()
                break
        
        # Industry indicators
        industry_keywords = [
            'technology', 'software', 'fintech', 'healthcare', 'e-commerce', 'education',
            'manufacturing', 'consulting', 'finance', 'banking', 'insurance', 'retail',
            'telecommunications', 'automotive', 'aerospace', 'energy', 'media', 'gaming'
        ]
        
        for industry in industry_keywords:
            if industry in text_lower:
                company_info['industry'] = industry.title()
                break
        
        # Company stage
        stage_indicators = [
            'startup', 'scale-up', 'established', 'fortune 500', 'unicorn', 'series a',
            'series b', 'series c', 'ipo', 'public company', 'private company'
        ]
        
        for stage in stage_indicators:
            if stage in text_lower:
                company_info['stage'] = stage.title()
                break
        
        return company_info
    
    def _extract_application_deadline(self, text: str) -> str:
        """Extract application deadline"""
        text_lower = text.lower()
        
        deadline_patterns = [
            r'deadline\s*:?\s*([^\n]+)',
            r'apply\s+by\s+([^\n]+)',
            r'last\s+date\s*:?\s*([^\n]+)',
            r'applications?\s+close\s+(?:on\s+)?([^\n]+)'
        ]
        
        for pattern in deadline_patterns:
            match = re.search(pattern, text_lower)
            if match:
                return match.group(1).strip()
        
        return ""
    
    def _extract_job_summary(self, text: str) -> str:
        """Extract job summary or overview"""
        text_lower = text.lower()
        
        summary_patterns = [
            r'(?:job\s+)?(?:summary|overview|description)\s*:?\s*(.*?)(?=(?:responsibilities|requirements|qualifications|skills|experience)|$)',
            r'(?:about\s+(?:the\s+)?(?:role|position|job))\s*:?\s*(.*?)(?=(?:responsibilities|requirements|qualifications|skills|experience)|$)',
            r'(?:role\s+overview|position\s+summary)\s*:?\s*(.*?)(?=(?:responsibilities|requirements|qualifications|skills|experience)|$)'
        ]
        
        for pattern in summary_patterns:
            match = re.search(pattern, text_lower, re.DOTALL)
            if match:
                summary = match.group(1).strip()
                # Clean up the summary
                summary = re.sub(r'\s+', ' ', summary)
                return summary[:300] + '...' if len(summary) > 300 else summary
        
        # If no dedicated summary, extract first paragraph
        paragraphs = text.split('\n\n')
        for para in paragraphs[:3]:
            para = para.strip()
            if len(para) > 100 and len(para) < 500:
                # Check if it looks like a summary (not a list)
                if not re.match(r'^\s*[•\*\-\d]', para):
                    return para[:300] + '...' if len(para) > 300 else para
        
        return ""
    
    def _calculate_job_complexity(self, parsed_jd: Dict) -> int:
        """Calculate job complexity score based on requirements"""
        complexity_score = 0
        
        # Required skills complexity (40%)
        required_skills = parsed_jd.get('required_skills', [])
        complexity_score += min(40, len(required_skills) * 3)
        
        # Experience requirements (25%)
        exp_req = parsed_jd.get('experience_required', {})
        min_years = exp_req.get('min_years', 0)
        complexity_score += min(25, min_years * 4)
        
        # Qualifications complexity (20%)
        qualifications = parsed_jd.get('qualifications', [])
        complexity_score += min(20, len(qualifications) * 4)
        
        # Responsibilities complexity (15%)
        responsibilities = parsed_jd.get('responsibilities', [])
        complexity_score += min(15, len(responsibilities) * 2)
        
        # Bonus for specific experience requirements
        specific_exp = exp_req.get('specific_experience', [])
        if specific_exp:
            complexity_score += min(10, len(specific_exp) * 2)
        
        return min(100, complexity_score)
    
    def get_job_priority_score(self, parsed_jd: Dict) -> int:
        """Calculate job priority score for HR dashboard"""
        priority_score = 0
        
        # High complexity jobs get higher priority
        complexity = parsed_jd.get('complexity_score', 0)
        priority_score += complexity * 0.3
        
        # Jobs with many requirements get higher priority
        total_requirements = (
            len(parsed_jd.get('required_skills', [])) +
            len(parsed_jd.get('qualifications', [])) +
            len(parsed_jd.get('responsibilities', []))
        )
        priority_score += min(30, total_requirements * 2)
        
        # Experience requirements
        exp_req = parsed_jd.get('experience_required', {})
        min_years = exp_req.get('min_years', 0)
        priority_score += min(20, min_years * 3)
        
        # Specific domain expertise requirements
        specific_exp = exp_req.get('specific_experience', [])
        priority_score += min(20, len(specific_exp) * 4)
        
        return min(100, int(priority_score))
    
    def extract_skill_categories(self, parsed_jd: Dict) -> Dict[str, List[str]]:
        """Categorize extracted skills by type"""
        categorized_skills = {}
        
        all_skills = parsed_jd.get('required_skills', []) + parsed_jd.get('preferred_skills', [])
        
        for skill in all_skills:
            skill_lower = skill.lower()
            categorized = False
            
            for category, skills_list in self.skills_database.items():
                if skill_lower in [s.lower() for s in skills_list]:
                    if category not in categorized_skills:
                        categorized_skills[category] = []
                    categorized_skills[category].append(skill)
                    categorized = True
                    break
            
            # If not categorized, add to general skills
            if not categorized:
                if 'other_skills' not in categorized_skills:
                    categorized_skills['other_skills'] = []
                categorized_skills['other_skills'].append(skill)
        
        return categorized_skills
    
    def get_job_matching_keywords(self, parsed_jd: Dict) -> List[str]:
        """Get prioritized keywords for resume matching"""
        keywords = []
        
        # Required skills have highest priority
        required_skills = parsed_jd.get('required_skills', [])
        keywords.extend([skill.lower() for skill in required_skills])
        
        # Add preferred skills with lower priority
        preferred_skills = parsed_jd.get('preferred_skills', [])
        keywords.extend([skill.lower() for skill in preferred_skills])
        
        # Add experience-related keywords
        exp_req = parsed_jd.get('experience_required', {})
        specific_exp = exp_req.get('specific_experience', [])
        for exp in specific_exp:
            # Extract key terms from experience descriptions
            exp_keywords = re.findall(r'\b[a-zA-Z]{3,}\b', exp.lower())
            keywords.extend(exp_keywords)
        
        # Add qualification keywords
        qualifications = parsed_jd.get('qualifications', [])
        for qual in qualifications:
            qual_keywords = re.findall(r'\b[a-zA-Z]{3,}\b', qual.lower())
            keywords.extend(qual_keywords)
        
        # Remove duplicates while preserving order
        seen = set()
        unique_keywords = []
        for keyword in keywords:
            if keyword not in seen and len(keyword) > 2:
                seen.add(keyword)
                unique_keywords.append(keyword)
        
        return unique_keywords[:50]  # Top 50 keywords
    
    def analyze_job_market_demand(self, parsed_jd: Dict) -> Dict:
        """Analyze job market demand based on skills and requirements"""
        analysis = {
            'demand_level': 'Medium',
            'skill_rarity': {},
            'market_insights': []
        }
        
        # Define high-demand skills
        high_demand_skills = [
            'python', 'javascript', 'react', 'aws', 'docker', 'kubernetes', 'machine learning',
            'data science', 'artificial intelligence', 'cloud computing', 'devops', 'agile',
            'microservices', 'api', 'sql', 'nosql', 'git', 'ci/cd'
        ]
        
        # Define rare/specialized skills
        rare_skills = [
            'blockchain', 'quantum computing', 'webassembly', 'rust', 'go', 'erlang',
            'haskell', 'scala', 'clojure', 'computer vision', 'nlp', 'deep learning',
            'reinforcement learning', 'edge computing', 'iot'
        ]
        
        required_skills = [skill.lower() for skill in parsed_jd.get('required_skills', [])]
        
        # Calculate demand level
        high_demand_count = sum(1 for skill in required_skills if skill in high_demand_skills)
        rare_skill_count = sum(1 for skill in required_skills if skill in rare_skills)
        
        if high_demand_count >= 3 or rare_skill_count >= 1:
            analysis['demand_level'] = 'High'
        elif high_demand_count >= 1:
            analysis['demand_level'] = 'Medium'
        else:
            analysis['demand_level'] = 'Low'
        
        # Analyze skill rarity
        for skill in required_skills:
            if skill in rare_skills:
                analysis['skill_rarity'][skill] = 'Rare'
            elif skill in high_demand_skills:
                analysis['skill_rarity'][skill] = 'High Demand'
            else:
                analysis['skill_rarity'][skill] = 'Common'
        
        # Generate market insights
        if rare_skill_count > 0:
            analysis['market_insights'].append("Position requires specialized/rare skills")
        
        if high_demand_count >= 3:
            analysis['market_insights'].append("High competition expected for this role")
        
        exp_req = parsed_jd.get('experience_required', {})
        min_years = exp_req.get('min_years', 0)
        if min_years >= 5:
            analysis['market_insights'].append("Senior-level position with high experience requirements")
        elif min_years <= 1:
            analysis['market_insights'].append("Entry-level or junior position")
        
        return analysis
    
    def get_similar_job_titles(self, job_title: str) -> List[str]:
        """Get similar job titles based on the given title"""
        title_lower = job_title.lower()
        
        similar_titles = []
        
        # Define job title mappings
        title_mappings = {
            'data scientist': ['data analyst', 'machine learning engineer', 'ai engineer', 'research scientist'],
            'software engineer': ['software developer', 'programmer', 'full stack developer', 'backend developer'],
            'frontend developer': ['ui developer', 'react developer', 'javascript developer', 'web developer'],
            'product manager': ['product owner', 'program manager', 'project manager', 'business analyst'],
            'devops engineer': ['site reliability engineer', 'cloud engineer', 'infrastructure engineer', 'platform engineer'],
            'data engineer': ['big data engineer', 'etl developer', 'data pipeline engineer', 'analytics engineer'],
            'ui/ux designer': ['product designer', 'visual designer', 'interaction designer', 'user researcher'],
            'business analyst': ['systems analyst', 'functional analyst', 'requirements analyst', 'process analyst'],
            'qa engineer': ['test engineer', 'sdet', 'automation engineer', 'quality analyst'],
            'marketing manager': ['digital marketing manager', 'growth manager', 'brand manager', 'content manager']
        }
        
        # Find similar titles
        for key, similar in title_mappings.items():
            if key in title_lower:
                similar_titles.extend(similar)
                break
        
        # If no direct match, try partial matches
        if not similar_titles:
            if 'engineer' in title_lower:
                similar_titles.extend(['developer', 'programmer', 'architect', 'specialist'])
            elif 'manager' in title_lower:
                similar_titles.extend(['lead', 'head', 'director', 'coordinator'])
            elif 'analyst' in title_lower:
                similar_titles.extend(['specialist', 'consultant', 'researcher', 'associate'])
        
        return similar_titles[:5]
    
    def validate_job_description_quality(self, parsed_jd: Dict) -> Dict:
        """Validate the quality and completeness of job description"""
        quality_report = {
            'score': 0,
            'issues': [],
            'suggestions': [],
            'completeness': {}
        }
        
        # Check required skills
        required_skills = parsed_jd.get('required_skills', [])
        if len(required_skills) >= 5:
            quality_report['score'] += 25
            quality_report['completeness']['skills'] = 'Good'
        elif len(required_skills) >= 2:
            quality_report['score'] += 15
            quality_report['completeness']['skills'] = 'Fair'
            quality_report['suggestions'].append("Consider adding more specific skill requirements")
        else:
            quality_report['completeness']['skills'] = 'Poor'
            quality_report['issues'].append("Very few or no specific skills mentioned")
        
        # Check experience requirements
        exp_req = parsed_jd.get('experience_required', {})
        if exp_req.get('min_years', 0) > 0:
            quality_report['score'] += 20
            quality_report['completeness']['experience'] = 'Good'
        else:
            quality_report['completeness']['experience'] = 'Poor'
            quality_report['issues'].append("No clear experience requirements specified")
        
        # Check responsibilities
        responsibilities = parsed_jd.get('responsibilities', [])
        if len(responsibilities) >= 5:
            quality_report['score'] += 20
            quality_report['completeness']['responsibilities'] = 'Good'
        elif len(responsibilities) >= 2:
            quality_report['score'] += 10
            quality_report['completeness']['responsibilities'] = 'Fair'
        else:
            quality_report['completeness']['responsibilities'] = 'Poor'
            quality_report['issues'].append("Job responsibilities not clearly defined")
        
        # Check qualifications
        qualifications = parsed_jd.get('qualifications', [])
        if len(qualifications) >= 2:
            quality_report['score'] += 15
            quality_report['completeness']['qualifications'] = 'Good'
        elif len(qualifications) >= 1:
            quality_report['score'] += 10
            quality_report['completeness']['qualifications'] = 'Fair'
        else:
            quality_report['completeness']['qualifications'] = 'Poor'
            quality_report['suggestions'].append("Add educational or certification requirements")
        
        # Check job summary
        job_summary = parsed_jd.get('job_summary', '')
        if len(job_summary) >= 100:
            quality_report['score'] += 10
            quality_report['completeness']['summary'] = 'Good'
        elif len(job_summary) >= 50:
            quality_report['score'] += 5
            quality_report['completeness']['summary'] = 'Fair'
        else:
            quality_report['completeness']['summary'] = 'Poor'
            quality_report['suggestions'].append("Add a comprehensive job summary or overview")
        
        # Check benefits
        benefits = parsed_jd.get('company_benefits', [])
        if len(benefits) >= 3:
            quality_report['score'] += 10
            quality_report['completeness']['benefits'] = 'Good'
        elif len(benefits) >= 1:
            quality_report['score'] += 5
            quality_report['completeness']['benefits'] = 'Fair'
        else:
            quality_report['completeness']['benefits'] = 'Poor'
            quality_report['suggestions'].append("Consider adding company benefits and perks")
        
        # Overall quality assessment
        if quality_report['score'] >= 80:
            quality_report['overall'] = 'Excellent'
        elif quality_report['score'] >= 60:
            quality_report['overall'] = 'Good'
        elif quality_report['score'] >= 40:
            quality_report['overall'] = 'Fair'
        else:
            quality_report['overall'] = 'Poor'
        
        return quality_report
