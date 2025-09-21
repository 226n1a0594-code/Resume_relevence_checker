import sqlite3
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import streamlit as st
import pandas as pd

class DatabaseManager:
    def __init__(self, db_path: str = "resume_system.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize database with enhanced schema"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Enhanced Job Descriptions table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS job_descriptions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                job_title TEXT NOT NULL,
                company_name TEXT NOT NULL,
                location TEXT NOT NULL,
                experience_level TEXT,
                job_type TEXT,
                priority TEXT DEFAULT 'Medium',
                raw_description TEXT NOT NULL,
                parsed_data TEXT NOT NULL,
                complexity_score INTEGER DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                is_active BOOLEAN DEFAULT 1,
                created_by TEXT DEFAULT 'system',
                application_deadline TEXT,
                salary_range TEXT,
                views_count INTEGER DEFAULT 0
            )
        ''')
        
        # Enhanced Evaluations table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS evaluations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                job_id INTEGER NOT NULL,
                candidate_name TEXT NOT NULL,
                candidate_email TEXT NOT NULL,
                candidate_phone TEXT,
                experience_years INTEGER DEFAULT 0,
                current_location TEXT,
                resume_text TEXT NOT NULL,
                parsed_resume TEXT NOT NULL,
                analysis_result TEXT NOT NULL,
                confidence_score INTEGER DEFAULT 0,
                resume_strength_score INTEGER DEFAULT 0,
                application_source TEXT DEFAULT 'direct',
                status TEXT DEFAULT 'new',
                notes TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (job_id) REFERENCES job_descriptions (id)
            )
        ''')
        
        # Analytics and Events table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS analytics_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                event_type TEXT NOT NULL,
                event_data TEXT,
                job_id INTEGER,
                evaluation_id INTEGER,
                session_id TEXT,
                ip_address TEXT,
                user_agent TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (job_id) REFERENCES job_descriptions (id),
                FOREIGN KEY (evaluation_id) REFERENCES evaluations (id)
            )
        ''')
        
        # System Configuration table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS system_config (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                config_key TEXT UNIQUE NOT NULL,
                config_value TEXT NOT NULL,
                description TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Interview Feedback table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS interview_feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                evaluation_id INTEGER NOT NULL,
                interviewer_name TEXT,
                interview_type TEXT,
                feedback_data TEXT,
                rating INTEGER,
                recommendation TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (evaluation_id) REFERENCES evaluations (id)
            )
        ''')
        
        # Create indexes for better performance
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_evaluations_job_id ON evaluations(job_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_evaluations_email ON evaluations(candidate_email)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_evaluations_created_at ON evaluations(created_at)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_job_descriptions_active ON job_descriptions(is_active)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_analytics_events_type ON analytics_events(event_type)')
        
        # Insert default configuration
        self._insert_default_config(cursor)
        
        conn.commit()
        conn.close()
    
    def _insert_default_config(self, cursor):
        """Insert default system configuration"""
        default_configs = [
            ('min_relevance_score', '50', 'Minimum relevance score for consideration'),
            ('auto_archive_days', '30', 'Days after which inactive jobs are archived'),
            ('max_file_size_mb', '5', 'Maximum file size for resume uploads in MB'),
            ('notification_email', 'hr@innomatics.com', 'Default notification email'),
            ('system_version', '2.0.0', 'Current system version'),
            ('analytics_retention_days', '365', 'Days to retain analytics data')
        ]
        
        for key, value, desc in default_configs:
            cursor.execute('''
                INSERT OR IGNORE INTO system_config (config_key, config_value, description)
                VALUES (?, ?, ?)
            ''', (key, value, desc))
    
    def save_job_description(self, jd_data: Dict) -> int:
        """Enhanced job description saving with metadata"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            # Calculate complexity score
            complexity_score = jd_data.get('parsed_data', {}).get('complexity_score', 0)
            
            cursor.execute('''
                INSERT INTO job_descriptions 
                (job_title, company_name, location, experience_level, job_type, 
                 priority, raw_description, parsed_data, complexity_score, application_deadline, salary_range)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                jd_data['job_title'],
                jd_data['company_name'],
                jd_data['location'],
                jd_data['experience_level'],
                jd_data['job_type'],
                jd_data['priority'],
                jd_data['raw_description'],
                json.dumps(jd_data['parsed_data']),
                complexity_score,
                jd_data.get('application_deadline', ''),
                json.dumps(jd_data.get('salary_range', {}))
            ))
            
            jd_id = cursor.lastrowid
            
            # Log analytics event
            self._log_analytics_event(cursor, 'jd_uploaded', {
                'job_title': jd_data['job_title'],
                'company': jd_data['company_name'],
                'complexity_score': complexity_score
            }, job_id=jd_id)
            
            conn.commit()
            return jd_id
            
        except Exception as e:
            conn.rollback()
            st.error(f"❌ Error saving job description: {str(e)}")
            return None
        finally:
            conn.close()
    
    def save_evaluation(self, eval_data: Dict) -> int:
        """Enhanced evaluation saving with additional metrics"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            # Extract additional metrics
            analysis_result = eval_data['analysis_result']
            confidence_score = analysis_result.get('confidence_score', 0)
            
            # Calculate resume strength score if not provided
            resume_strength_score = eval_data.get('resume_strength_score', 0)
            
            cursor.execute('''
                INSERT INTO evaluations 
                (job_id, candidate_name, candidate_email, candidate_phone,
                 experience_years, current_location, resume_text, parsed_resume, 
                 analysis_result, confidence_score, resume_strength_score, application_source)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                eval_data['job_id'],
                eval_data['candidate_name'],
                eval_data['candidate_email'],
                eval_data.get('candidate_phone'),
                eval_data.get('experience_years', 0),
                eval_data.get('current_location'),
                eval_data['resume_text'],
                json.dumps(eval_data['parsed_resume']),
                json.dumps(eval_data['analysis_result']),
                confidence_score,
                resume_strength_score,
                eval_data.get('application_source', 'direct')
            ))
            
            eval_id = cursor.lastrowid
            
            # Log analytics event
            self._log_analytics_event(cursor, 'resume_evaluated', {
                'job_id': eval_data['job_id'],
                'score': analysis_result['relevance_score'],
                'verdict': analysis_result['verdict'],
                'confidence_score': confidence_score
            }, job_id=eval_data['job_id'], evaluation_id=eval_id)
            
            conn.commit()
            return eval_id
            
        except Exception as e:
            conn.rollback()
            st.error(f"❌ Error saving evaluation: {str(e)}")
            return None
        finally:
            conn.close()
    
    def get_active_job_descriptions(self) -> List[Tuple]:
        """Get all active job descriptions with enhanced data"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute('''
                SELECT 
                    id, job_title, company_name, location, created_at, 
                    complexity_score, priority, experience_level,
                    (SELECT COUNT(*) FROM evaluations WHERE job_id = job_descriptions.id) as application_count
                FROM job_descriptions 
                WHERE is_active = 1
                ORDER BY created_at DESC
            ''')
            
            return cursor.fetchall()
            
        except Exception as e:
            st.error(f"❌ Error fetching job descriptions: {str(e)}")
            return []
        finally:
            conn.close()
    
    def get_job_description(self, job_id: int) -> Dict:
        """Get specific job description with metadata"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute('''
                SELECT 
                    job_title, company_name, location, raw_description, parsed_data,
                    complexity_score, priority, experience_level, application_deadline,
                    salary_range, views_count
                FROM job_descriptions 
                WHERE id = ?
            ''', (job_id,))
            
            result = cursor.fetchone()
            if result:
                # Increment view count
                cursor.execute('UPDATE job_descriptions SET views_count = views_count + 1 WHERE id = ?', (job_id,))
                conn.commit()
                
                return {
                    'job_title': result[0],
                    'company_name': result[1],
                    'location': result[2],
                    'raw_description': result[3],
                    'parsed_data': json.loads(result[4]),
                    'complexity_score': result[5],
                    'priority': result[6],
                    'experience_level': result[7],
                    'application_deadline': result[8],
                    'salary_range': json.loads(result[9] or '{}'),
                    'views_count': result[10]
                }
            return None
            
        except Exception as e:
            st.error(f"❌ Error fetching job description: {str(e)}")
            return None
        finally:
            conn.close()
    
    def get_dashboard_stats(self) -> Dict:
        """Get comprehensive dashboard statistics"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            stats = {}
            
            # Basic counts
            cursor.execute('SELECT COUNT(*) FROM job_descriptions WHERE is_active = 1')
            stats['total_jds'] = cursor.fetchone()[0]
            
            cursor.execute('SELECT COUNT(DISTINCT candidate_email) FROM evaluations')
            stats['total_resumes'] = cursor.fetchone()[0]
            
            cursor.execute('SELECT COUNT(*) FROM evaluations')
            stats['total_evaluations'] = cursor.fetchone()[0]
            
            cursor.execute('''
                SELECT COUNT(*) FROM evaluations 
                WHERE created_at >= date('now', '-7 days')
            ''')
            stats['weekly_applications'] = cursor.fetchone()[0]
            
            # Quality metrics
            cursor.execute('''
                SELECT 
                    AVG(CAST(json_extract(analysis_result, '$.relevance_score') AS INTEGER)) as avg_score,
                    COUNT(CASE WHEN json_extract(analysis_result, '$.verdict') = 'High' THEN 1 END) as high_quality,
                    COUNT(CASE WHEN json_extract(analysis_result, '$.verdict') = 'Medium' THEN 1 END) as medium_quality,
                    COUNT(CASE WHEN json_extract(analysis_result, '$.verdict') = 'Low' THEN 1 END) as low_quality
                FROM evaluations
            ''')
            
            quality_result = cursor.fetchone()
            if quality_result:
                stats['average_score'] = round(quality_result[0] or 0, 1)
                stats['high_quality_candidates'] = quality_result[1]
                stats['medium_quality_candidates'] = quality_result[2]
                stats['low_quality_candidates'] = quality_result[3]
            
            # Trending data
            cursor.execute('''
                SELECT DATE(created_at) as date, COUNT(*) as count
                FROM evaluations
                WHERE created_at >= date('now', '-30 days')
                GROUP BY DATE(created_at)
                ORDER BY date DESC
                LIMIT 7
            ''')
            
            stats['weekly_trend'] = cursor.fetchall()
            
            # Top performing jobs
            cursor.execute('''
                SELECT 
                    j.job_title,
                    COUNT(e.id) as application_count,
                    AVG(CAST(json_extract(e.analysis_result, '$.relevance_score') AS INTEGER)) as avg_score
                FROM job_descriptions j
                LEFT JOIN evaluations e ON j.id = e.job_id
                WHERE j.is_active = 1
                GROUP BY j.id, j.job_title
                HAVING application_count > 0
                ORDER BY application_count DESC
                LIMIT 5
            ''')
            
            stats['top_jobs'] = cursor.fetchall()
            
            return stats
            
        except Exception as e:
            st.error(f"❌ Error fetching dashboard stats: {str(e)}")
            return {
                'total_jds': 0, 'total_resumes': 0, 'total_evaluations': 0,
                'weekly_applications': 0, 'average_score': 0, 'high_quality_candidates': 0,
                'medium_quality_candidates': 0, 'low_quality_candidates': 0,
                'weekly_trend': [], 'top_jobs': []
            }
        finally:
            conn.close()
    
    def get_recent_evaluations(self, limit: int = 10) -> List[Tuple]:
        """Get recent evaluations with enhanced data"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute('''
                SELECT 
                    e.id,
                    j.job_title,
                    e.candidate_name,
                    CAST(json_extract(e.analysis_result, '$.relevance_score') AS INTEGER) as score,
                    json_extract(e.analysis_result, '$.verdict') as verdict,
                    e.created_at,
                    e.confidence_score,
                    e.status
                FROM evaluations e
                JOIN job_descriptions j ON e.job_id = j.id
                ORDER BY e.created_at DESC
                LIMIT ?
            ''', (limit,))
            
            return cursor.fetchall()
            
        except Exception as e:
            st.error(f"❌ Error fetching recent evaluations: {str(e)}")
            return []
        finally:
            conn.close()
    
    def get_evaluations_filtered(self, job_filter: str = "All Jobs", 
                                verdict_filter: str = "All", 
                                min_score: int = 0,
                                date_range: Tuple[str, str] = None) -> List[Tuple]:
        """Get filtered evaluations with advanced filtering"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            base_query = '''
                SELECT 
                    e.id,
                    j.job_title,
                    e.candidate_name,
                    e.candidate_email,
                    CAST(json_extract(e.analysis_result, '$.relevance_score') AS INTEGER) as score,
                    json_extract(e.analysis_result, '$.verdict') as verdict,
                    e.created_at,
                    e.current_location,
                    e.experience_years,
                    e.confidence_score,
                    e.status
                FROM evaluations e
                JOIN job_descriptions j ON e.job_id = j.id
                WHERE 1=1
            '''
            
            params = []
            
            if job_filter != "All Jobs":
                base_query += " AND j.job_title = ?"
                params.append(job_filter)
            
            if verdict_filter != "All":
                base_query += " AND json_extract(e.analysis_result, '$.verdict') = ?"
                params.append(verdict_filter)
            
            base_query += " AND CAST(json_extract(e.analysis_result, '$.relevance_score') AS INTEGER) >= ?"
            params.append(min_score)
            
            if date_range:
                base_query += " AND e.created_at BETWEEN ? AND ?"
                params.extend(date_range)
            
            base_query += " ORDER BY CAST(json_extract(e.analysis_result, '$.relevance_score') AS INTEGER) DESC"
            
            cursor.execute(base_query, params)
            return cursor.fetchall()
            
        except Exception as e:
            st.error(f"❌ Error fetching filtered evaluations: {str(e)}")
            return []
        finally:
            conn.close()
    
    def get_detailed_evaluation(self, eval_id: int) -> Dict:
        """Get detailed evaluation with full analysis"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute('''
                SELECT 
                    e.candidate_name, e.candidate_email, e.candidate_phone,
                    e.experience_years, e.current_location,
                    j.job_title, j.company_name, j.location as job_location,
                    e.parsed_resume, e.analysis_result, e.confidence_score,
                    e.resume_strength_score, e.status, e.notes,
                    e.created_at, e.updated_at
                FROM evaluations e
                JOIN job_descriptions j ON e.job_id = j.id
                WHERE e.id = ?
            ''', (eval_id,))
            
            result = cursor.fetchone()
            if result:
                return {
                    'candidate_name': result[0],
                    'candidate_email': result[1],
                    'candidate_phone': result[2],
                    'experience_years': result[3],
                    'current_location': result[4],
                    'job_title': result[5],
                    'company_name': result[6],
                    'job_location': result[7],
                    'parsed_resume': json.loads(result[8]),
                    'analysis_result': json.loads(result[9]),
                    'confidence_score': result[10],
                    'resume_strength_score': result[11],
                    'status': result[12],
                    'notes': result[13],
                    'created_at': result[14],
                    'updated_at': result[15]
                }
            return None
            
        except Exception as e:
            st.error(f"❌ Error fetching detailed evaluation: {str(e)}")
            return None
        finally:
            conn.close()
    
    def get_analytics_data(self) -> Dict:
        """Get comprehensive analytics data"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            analytics = {}
            
            # Get all evaluations for analytics
            cursor.execute('''
                SELECT 
                    e.id, j.job_title, e.candidate_name, e.candidate_email,
                    CAST(json_extract(e.analysis_result, '$.relevance_score') AS INTEGER) as score,
                    json_extract(e.analysis_result, '$.verdict') as verdict,
                    e.created_at, e.experience_years, e.current_location,
                    e.confidence_score, j.complexity_score, j.priority
                FROM evaluations e
                JOIN job_descriptions j ON e.job_id = j.id
                ORDER BY e.created_at DESC
            ''')
            
            analytics['evaluations'] = cursor.fetchall()
            
            # Get job posting trends
            cursor.execute('''
                SELECT 
                    DATE(created_at) as date, 
                    COUNT(*) as count,
                    AVG(complexity_score) as avg_complexity
                FROM job_descriptions
                WHERE created_at >= date('now', '-90 days')
                GROUP BY DATE(created_at)
                ORDER BY date
            ''')
            
            analytics['job_trends'] = cursor.fetchall()
            
            # Get skill demand analysis
            cursor.execute('''
                SELECT 
                    j.job_title,
                    j.parsed_data,
                    COUNT(e.id) as application_count,
                    AVG(CAST(json_extract(e.analysis_result, '$.relevance_score') AS INTEGER)) as avg_score
                FROM job_descriptions j
                LEFT JOIN evaluations e ON j.id = e.job_id
                WHERE j.is_active = 1
                GROUP BY j.id
                HAVING application_count > 0
            ''')
            
            analytics['skill_demand'] = cursor.fetchall()
            
            # Get performance metrics by time
            cursor.execute('''
                SELECT 
                    strftime('%Y-%m', e.created_at) as month,
                    COUNT(*) as total_applications,
                    COUNT(CASE WHEN json_extract(e.analysis_result, '$.verdict') = 'High' THEN 1 END) as high_quality,
                    AVG(CAST(json_extract(e.analysis_result, '$.relevance_score') AS INTEGER)) as avg_score
                FROM evaluations e
                WHERE e.created_at >= date('now', '-12 months')
                GROUP BY strftime('%Y-%m', e.created_at)
                ORDER BY month
            ''')
            
            analytics['monthly_performance'] = cursor.fetchall()
            
            return analytics
            
        except Exception as e:
            st.error(f"❌ Error fetching analytics data: {str(e)}")
            return {
                'evaluations': [], 'job_trends': [], 
                'skill_demand': [], 'monthly_performance': []
            }
        finally:
            conn.close()
    
    def update_evaluation_status(self, eval_id: int, status: str, notes: str = None) -> bool:
        """Update evaluation status and notes"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute('''
                UPDATE evaluations 
                SET status = ?, notes = ?, updated_at = CURRENT_TIMESTAMP
                WHERE id = ?
            ''', (status, notes, eval_id))
            
            success = cursor.rowcount > 0
            
            if success:
                self._log_analytics_event(cursor, 'status_updated', {
                    'evaluation_id': eval_id,
                    'new_status': status
                }, evaluation_id=eval_id)
            
            conn.commit()
            return success
            
        except Exception as e:
            st.error(f"❌ Error updating evaluation status: {str(e)}")
            return False
        finally:
            conn.close()
    
    def get_candidate_history(self, candidate_email: str) -> List[Dict]:
        """Get comprehensive application history for a candidate"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute('''
                SELECT 
                    j.job_title, j.company_name, j.location,
                    CAST(json_extract(e.analysis_result, '$.relevance_score') AS INTEGER) as score,
                    json_extract(e.analysis_result, '$.verdict') as verdict,
                    e.created_at, e.status, e.confidence_score
                FROM evaluations e
                JOIN job_descriptions j ON e.job_id = j.id
                WHERE e.candidate_email = ?
                ORDER BY e.created_at DESC
            ''', (candidate_email,))
            
            results = cursor.fetchall()
            
            history = []
            for result in results:
                history.append({
                    'job_title': result[0],
                    'company_name': result[1],
                    'location': result[2],
                    'score': result[3],
                    'verdict': result[4],
                    'applied_date': result[5],
                    'status': result[6],
                    'confidence_score': result[7]
                })
            
            return history
            
        except Exception as e:
            st.error(f"❌ Error fetching candidate history: {str(e)}")
            return []
        finally:
            conn.close()
    
    def get_top_candidates_for_job(self, job_id: int, limit: int = 10) -> List[Dict]:
        """Get top candidates for a specific job with enhanced metrics"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute('''
                SELECT 
                    e.id, e.candidate_name, e.candidate_email, e.experience_years,
                    e.current_location, e.candidate_phone,
                    CAST(json_extract(e.analysis_result, '$.relevance_score') AS INTEGER) as score,
                    json_extract(e.analysis_result, '$.verdict') as verdict,
                    e.analysis_result, e.confidence_score, e.resume_strength_score,
                    e.status, e.created_at
                FROM evaluations e
                WHERE e.job_id = ?
                ORDER BY 
                    CAST(json_extract(e.analysis_result, '$.relevance_score') AS INTEGER) DESC,
                    e.confidence_score DESC
                LIMIT ?
            ''', (job_id, limit))
            
            results = cursor.fetchall()
            
            candidates = []
            for result in results:
                candidates.append({
                    'evaluation_id': result[0],
                    'name': result[1],
                    'email': result[2],
                    'experience_years': result[3],
                    'location': result[4],
                    'phone': result[5],
                    'score': result[6],
                    'verdict': result[7],
                    'full_analysis': json.loads(result[8]),
                    'confidence_score': result[9],
                    'resume_strength_score': result[10],
                    'status': result[11],
                    'applied_date': result[12]
                })
            
            return candidates
            
        except Exception as e:
            st.error(f"❌ Error fetching top candidates: {str(e)}")
            return []
        finally:
            conn.close()
    
    def _log_analytics_event(self, cursor, event_type: str, event_data: Dict, 
                           job_id: int = None, evaluation_id: int = None):
        """Log analytics event"""
        try:
            cursor.execute('''
                INSERT INTO analytics_events 
                (event_type, event_data, job_id, evaluation_id, session_id)
                VALUES (?, ?, ?, ?, ?)
            ''', (
                event_type, 
                json.dumps(event_data), 
                job_id, 
                evaluation_id,
                st.session_state.get('session_id', 'unknown')
            ))
        except Exception:
            # Silently fail for analytics to not disrupt main functionality
            pass
    
    def get_system_health(self) -> Dict:
        """Get system health and performance metrics"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            health = {}
            
            # Database size
            cursor.execute("SELECT page_count * page_size as size FROM pragma_page_count(), pragma_page_size()")
            db_size = cursor.fetchone()[0]
            health['database_size_mb'] = round(db_size / (1024 * 1024), 2)
            
            # Record counts
            cursor.execute('SELECT COUNT(*) FROM job_descriptions')
            health['total_jobs'] = cursor.fetchone()[0]
            
            cursor.execute('SELECT COUNT(*) FROM evaluations')
            health['total_evaluations'] = cursor.fetchone()[0]
            
            cursor.execute('SELECT COUNT(*) FROM analytics_events')
            health['total_events'] = cursor.fetchone()[0]
            
            # Recent activity
            cursor.execute('SELECT COUNT(*) FROM evaluations WHERE created_at >= datetime("now", "-24 hours")')
            health['evaluations_last_24h'] = cursor.fetchone()[0]
            
            # Average processing time (simulated)
            health['avg_processing_time_seconds'] = 2.3
            
            # System status
            if health['evaluations_last_24h'] > 0:
                health['status'] = 'Active'
            elif health['total_evaluations'] > 0:
                health['status'] = 'Idle'
            else:
                health['status'] = 'New'
            
            return health
            
        except Exception as e:
            return {
                'status': 'Error',
                'error_message': str(e),
                'database_size_mb': 0,
                'total_jobs': 0,
                'total_evaluations': 0,
                'total_events': 0,
                'evaluations_last_24h': 0,
                'avg_processing_time_seconds': 0
            }
        finally:
            conn.close()
    
    def cleanup_old_data(self, days_to_keep: int = 365) -> Dict:
        """Clean up old analytics data and archived records"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cutoff_date = datetime.now() - timedelta(days=days_to_keep)
            
            # Clean old analytics events
            cursor.execute('''
                DELETE FROM analytics_events 
                WHERE created_at < ?
            ''', (cutoff_date,))
            
            deleted_events = cursor.rowcount
            
            # Archive old evaluations for inactive jobs
            cursor.execute('''
                UPDATE evaluations 
                SET status = 'archived' 
                WHERE job_id IN (
                    SELECT id FROM job_descriptions 
                    WHERE is_active = 0 AND created_at < ?
                ) AND status != 'archived'
            ''', (cutoff_date,))
            
            archived_evaluations = cursor.rowcount
            
            conn.commit()
            
            return {
                'success': True,
                'deleted_events': deleted_events,
                'archived_evaluations': archived_evaluations
            }
            
        except Exception as e:
            conn.rollback()
            return {
                'success': False,
                'error': str(e)
            }
        finally:
            conn.close()
    
    def export_data_for_analysis(self, format_type: str = 'json') -> Dict:
        """Export data in various formats for external analysis"""
        conn = sqlite3.connect(self.db_path)
        
        try:
            # Get comprehensive data
            evaluations_df = pd.read_sql_query('''
                SELECT 
                    e.*, j.job_title, j.company_name, j.complexity_score,
                    CAST(json_extract(e.analysis_result, '$.relevance_score') AS INTEGER) as relevance_score,
                    json_extract(e.analysis_result, '$.verdict') as verdict
                FROM evaluations e
                JOIN job_descriptions j ON e.job_id = j.id
            ''', conn)
            
            jobs_df = pd.read_sql_query('SELECT * FROM job_descriptions', conn)
            
            if format_type == 'csv':
                return {
                    'evaluations_csv': evaluations_df.to_csv(index=False),
                    'jobs_csv': jobs_df.to_csv(index=False)
                }
            elif format_type == 'json':
                return {
                    'evaluations': evaluations_df.to_dict('records'),
                    'jobs': jobs_df.to_dict('records')
                }
            else:
                return {
                    'evaluations_df': evaluations_df,
                    'jobs_df': jobs_df
                }
                
        except Exception as e:
            return {'error': str(e)}
        finally:
            conn.close()