from pydantic import BaseModel, EmailStr, Field
from typing import Optional

class Student(BaseModel):
    name: str = 'Abhishek'
    age: Optional[int] = None
    email: EmailStr
    cgpa: float = Field(gt=0, lt=10, description= 'A decimal value to represent the grade of a student')

new_student= {'age': 98, 'email': 'abcdghbdh@gmail.com', 'cgpa': 9.9}  # here if we pass any integral or other data type value  then "pydantic" will show error (Thus, it handles the problem that was arising in the case of typeddict)
student = Student(**new_student)

# print(student)

dict_student = dict(student)

print(dict_student['age'])

json_student = student.model_dump()

print(json_student["email"])