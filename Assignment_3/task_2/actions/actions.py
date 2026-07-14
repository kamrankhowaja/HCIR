from typing import Any, Text, Dict, List
from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher
import requests
from datetime import date
from html.parser import HTMLParser
from html import unescape
import re


MENU_URL = "https://www.studierendenwerk-bonn.de/essen-trinken/mensen-cafes/mensa-sankt-augustin/"
MAX_MEALS = 6


def _clean_lines(text: Text) -> List[Text]:
    return [line.strip() for line in text.splitlines() if line.strip()]


def _unique(items: List[Text]) -> List[Text]:
    seen = set()
    unique_items = []
    for item in items:
        normalized = " ".join(item.split())
        if normalized and normalized.lower() not in seen:
            seen.add(normalized.lower())
            unique_items.append(normalized)
    return unique_items


class MensaMenuParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self._capture_link = False
        self._current_href = ""
        self._current_text: List[Text] = []
        self._capture_meal = False
        self._meal_depth = 0
        self._meal_text: List[Text] = []
        self.pdf_links: List[Text] = []
        self.meals: List[Text] = []

    def handle_starttag(self, tag: Text, attrs: List[tuple]) -> None:
        attributes = dict(attrs)
        classes = attributes.get("class", "")
        data_meal = any(name.startswith("data-meal") for name, _ in attrs)

        if tag == "a" and ".pdf" in attributes.get("href", "").lower():
            self._capture_link = True
            self._current_href = attributes["href"]
            self._current_text = []

        if self._capture_meal:
            self._meal_depth += 1
        elif data_meal or "meal-item" in classes or "meal-title" in classes:
            self._capture_meal = True
            self._meal_depth = 1
            self._meal_text = []

    def handle_data(self, data: Text) -> None:
        if self._capture_link:
            self._current_text.append(data)
        if self._capture_meal:
            self._meal_text.append(data)

    def handle_endtag(self, tag: Text) -> None:
        if tag == "a" and self._capture_link:
            text = " ".join(" ".join(self._current_text).split())
            if "Speiseplan" in text:
                self.pdf_links.append(f"{text}: {self._current_href}")
            self._capture_link = False
            self._current_href = ""
            self._current_text = []
        if self._capture_meal:
            self._meal_depth -= 1
            if self._meal_depth <= 0:
                text = " ".join(" ".join(self._meal_text).split())
                if text and not text.lower().startswith(("allergene", "zusatzstoffe")):
                    self.meals.append(text)
                self._capture_meal = False
                self._meal_text = []


def _parse_menu_page(html: Text) -> MensaMenuParser:
    parser = MensaMenuParser()
    parser.feed(html)
    parser.close()
    return parser


def _extract_pdf_links(html: Text) -> List[Text]:
    links = []
    pattern = re.compile(r'<a[^>]+href=["\']([^"\']*\.pdf[^"\']*)["\'][^>]*>(.*?)</a>', re.IGNORECASE | re.DOTALL)
    for href, label in pattern.findall(html):
        text = re.sub(r"<[^>]+>", " ", label)
        text = " ".join(unescape(text).split()) or "Speiseplan PDF"
        if "speiseplan" in text.lower():
            links.append(f"{text}: {unescape(href)}")
    return _unique(links)


def _requested_day(message: Text) -> Text:
    text = message.lower()
    if "tomorrow" in text or "morgen" in text:
        return "tomorrow"
    if "yesterday" in text or "gestern" in text:
        return "yesterday"
    return "today"


class ActionGetLunchMenu(Action):

    def name(self) -> Text:
        return "action_get_lunch_menu"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        }
        requested_day = _requested_day(tracker.latest_message.get("text", ""))

        try:
            response = requests.get(MENU_URL, headers=headers, timeout=10)
            if response.status_code != 200:
                dispatcher.utter_message(text="I couldn't reach the Studierendenwerk server right now. Please try again later.")
                return []

            menu_page = _parse_menu_page(response.text)
            meals = _unique(menu_page.meals)[:MAX_MEALS]

            if meals:
                menu_text = "\n- ".join(meals)
                dispatcher.utter_message(
                    text=f"Menu for {requested_day} at Mensa Sankt Augustin ({date.today().isoformat()}):\n- {menu_text}"
                )
                return []

            pdf_links = _unique(menu_page.pdf_links + _extract_pdf_links(response.text))
            if pdf_links:
                link_text = "\n".join(pdf_links[:2])
                day_note = ""
                if requested_day == "yesterday":
                    day_note = " Yesterday's menu may no longer be available on the live page."
                elif requested_day == "tomorrow":
                    day_note = " For tomorrow, use the current or next-week PDF depending on the date."
                dispatcher.utter_message(
                    text=(
                        f"I reached the live Mensa Sankt Augustin page, but the meals for {requested_day} are loaded dynamically and were not visible to my parser."
                        f"{day_note} "
                        f"You can open the current official menu here:\n{link_text}"
                    )
                )
                return []

            dispatcher.utter_message(
                text=(
                    f"The canteen page loaded, but I could not find the dishes for {requested_day}. "
                    f"Please check the live menu page: {MENU_URL}"
                )
            )

        except requests.RequestException:
            dispatcher.utter_message(text="I couldn't reach the Studierendenwerk server right now. Please try again later.")
        except Exception:
            dispatcher.utter_message(text="I ran into an issue parsing the live menu website. The canteen might be closed, or the site layout changed.")
            
        return []
