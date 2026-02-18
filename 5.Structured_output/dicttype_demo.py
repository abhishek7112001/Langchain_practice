from typing import TypedDict

class Person(TypedDict):
    name: str
    age: int

person: Person = {'name': 'Abhi', 'age': 22}

print(person)