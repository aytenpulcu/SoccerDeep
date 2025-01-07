# -*- coding: utf-8 -*-
"""
Created on Mon Dec 30 16:52:15 2024

@author: ayten
"""

from typing import List
from typing import Any
from dataclasses import dataclass

@dataclass
class Annotation:
    gameTime: str
    label: str
    position: str
    team: str
    visibility: str

    @staticmethod
    def from_dict(obj: Any) -> 'Annotation':
        _gameTime = str(obj.get("gameTime"))
        _label = str(obj.get("label"))
        _position = str(obj.get("position"))
        _team = str(obj.get("team"))
        _visibility = str(obj.get("visibility"))
        return Annotation(_gameTime, _label, _position, _team, _visibility)

@dataclass
class Root:
    UrlLocal: str
    UrlYoutube: str
    annotations: List[Annotation]

    @staticmethod
    def from_dict(obj: Any) -> 'Root':
        _UrlLocal = str(obj.get("UrlLocal"))
        _UrlYoutube = str(obj.get("UrlYoutube"))
        _annotations = [Annotation.from_dict(y) for y in obj.get("annotations")]
        return Root(_UrlLocal, _UrlYoutube, _annotations)
# Ana sınıflar
labels = [
    "PASS", "DRIVE", "HEADER", "HIGH PASS", "OUT", "CROSS", 
    "THROW IN", "SHOT", "BALL PLAYER BLOCK", "PLAYER SUCCESSFUL TACKLE", 
    "FREE KICK", "GOAL", "NO_ACTION"
]

# Veri yükleme ve model eğitimi
Train_video_path = "/Users/ayten/Documents/SoccerNet/spotting-ball-2024/train//england_efl/2019-2020/"
videos=["2019-10-01 - Blackburn Rovers - Nottingham Forest" ,
        "2019-10-01 - Brentford - Bristol City",
        "2019-10-01 - Hull City - Sheffield Wednesday",
        "2019-10-01 - Leeds United - West Bromwich",
        "2019-10-01 - Middlesbrough - Preston North End",
        "2019-10-01 - Reading - Fulham",
        "2019-10-01 - Stoke City - Huddersfield Town"]
