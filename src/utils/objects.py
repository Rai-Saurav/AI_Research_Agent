from pydantic import BaseModel , Field 
from typing import List


# creating our object Analyst

class Analyst(BaseModel):
    affiliation:str=Field(description='primary affiliation of the analyst.')
    name: str=Field(description="Nmae of the analyst")
    role: str=Field(description='Role of the analyst in the context of the topic')
    description:str=Field(description="description of the analyst focus , concerns, and motives")


    @property 
    def persona(self) -> str:
        return f"Name : {self.name}\nRole:{self.role}\nAffiliation:{self.affiliation}\nDescription : {self.description}"


class Perspectives(BaseModel):
    analysts:List[Analyst]=Field(description="comprehensive list of analysts with their roles and affilitions")

class SearchQuery(BaseModel):
    search_query: str=Field(description="search query for the retrieval"
                            )