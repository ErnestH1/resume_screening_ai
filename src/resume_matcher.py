#handle job descritpion and resume matching logic using semantic similarity

import pandas as pd
import numpy as np
from typing import List,Dict,Tuple,Optional
from .nlp_processor import NLPProcessor
import logging

#setup logging
logging.basicConfig(level=logging.INFO)
logger=logging.getLogger(__name__)

#create the resume matcher class
#using sematic similarity 

class ResumeMatcher:
    def __init__(self,nlp_processor:Optional[NLPProcessor]=None):
        #an instance of NLP processor
        self.nlp_processor=nlp_processor or NLPProcessor()
        self.job_description=None
        self.job_processed=None
        self.resumes=[]
        self.results=[]

    def set_job_descriptions(self,job_text:str)->Dict:
        logger.info('Processing job Description')
        self.job_description=job_text
        self.job_processed=self.nlp_processor(job_text)
        logger.info("JOb description processed successfully!")
        return self.job_processed
    
    def add_resume(self,resume_text:str,filename:str='')->Dict:
        #add a resume ,process ,output as processed resume
        existing_filenames=[r.get('filename', '') for r in self.resumes]
        if filename in existing_filenames:
            #return the existing resume
            for resume in self.resumes:
                if resume.get('filename')==filename:
                    return resume 
        #processing of a new resume 
        resume_processed=self.nlp_processor.process_document(resume_text)
        resume_processed['filename']=filename
        self.resumes.append(resume_processed)
        return resume_processed
    
    def calculate_semantic_similarity(self,resume_data:Dict)->float:
        if self.job_processed is None:
            raise ValueError('Job descriotion is not set ')
        job_embeddings=self.job_processed['embeddings']
        resume_embeddings=resume_data['embeddings']

        if job_embeddings.size==0 or resume_embeddings.size==0:
            return 0.0
        
        similarity= self.nlp_processor.calculate_similarity(
            job_embeddings.reshape(1,-1),
            resume_embeddings.reshape(1,-1)
        )

        return similarity
    
    def calculate_skill_match(self,resume_data:Dict)->Tuple[float,Dict]:
        if self.job_processed is None:
            raise ValueError("Job description not set")
        job_skills=self.job_processed['skills']
        resume_skils=resume_data['skills']

        #flatten the skills
        job_skills_flat=[]
        for skills_list in job_skills.values():
            job_skills_flat.extend(skills_list)
        resume_skills_flat=[]
        for skills_list in resume_skils.values():
            resume_skills_flat.extend(skills_list)

        if not job_skills_flat:
            return 0.5,{'matched_skills':[],'missing_skills':[],'match_percetage':0}
        
        #calculate skill matches
        matched_skills=list(set(job_skills_flat) & set(resume_skills_flat))
        missing_skills=list(set(job_skills_flat) - set(resume_skills_flat))

        #calculate the score
        match_percentage=len(matched_skills)/len(job_skills_flat) if job_skills_flat else 0

        #bonus for having additional skills
        bonus=min(0.2,len(resume_skills_flat)*0.02)
        skill_score=match_percentage + bonus
        skill_score=min(1.0,skill_score)

        skill_details={
            'matched_skills':matched_skills,
            'missing_skills':missing_skills,
            'match_percentage':match_percentage,
            'total_job_skills':len(job_skills_flat),
            'total_resume_skills':len(resume_skills_flat)
        }

        return skill_score,skill_details
    
    def calculate_experience_score(self,resume_data:Dict)->Tuple[float,Dict]:
        #calculate the experience match
        job_experience=self.job_processed.get('experience_years',0)
        resume_experience=self.resume_data.get('experience_years',0)

        if job_experience is None or resume_experience is None: