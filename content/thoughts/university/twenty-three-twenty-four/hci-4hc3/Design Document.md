---
id: Design Document
tags:
  - sfwr4hc3
date: "2024-01-07"
draft: true
noindex: true
title: Design Document
---

## Stakeholders

#### Students Seeking Quiet Study Areas

- **Characteristic Types**: Needs quiet, individual study time, possibly with accessibility requirements.
- **Priority Level**: Primary
- **Justification**: These students rely on the app to find environments conducive to individual study, which is a core purpose of the app. Their satisfaction directly impacts the app's success.
- **Information to Gather**:
  - Preferred devices for app use
  - Study habits and times
  - Importance of silence and amenities (like outlets)

#### Group Study Students

- **Characteristic Types**: Engaged in collaborative study sessions, require larger spaces, and may need to book spaces in advance.
- **Priority Level**: Primary
- **Justification**: Group study users frequently utilize campus facilities and will benefit from features that support group collaboration, making them a critical user group.
- **Information to Gather**:
  - Frequency of group study sessions
  - Typical group size and space requirements
  - Preferences for on-campus vs. off-campus study locations

#### Commuter Students

- **Characteristic Types**: Typically off-campus, prioritize study spots based on proximity and available time between classes.
- **Priority Level**: Secondary
- **Justification**: While they may not be on campus as much as residents, their need for efficient study spot utilization is high when they are on campus.
- **Information to Gather**:
  - Travel patterns and times
  - Preference for study spots near campus transit points
  - Usage of on-the-go mobile devices vs. stationary computing

#### Technologically Inclined Students

- **Characteristic Types**: Require high-tech amenities for their study sessions, such as fast Wi-Fi, power sources, and digital tools.
- **Priority Level**: Secondary
- **Justification**: These students represent a user group that is likely to contribute to the app's content by adding new tech-friendly study spots.
- **Information to Gather**:
  - Importance of tech amenities in a study place
  - Frequency of app usage to find and add new study spots
  - Interaction with app features for community engagement

#### Faculty and Staff

- **Characteristic Types**: Occasionally use study spaces for meetings or work outside the office, may recommend spaces to students.
- **Priority Level**: Tertiary
- **Justification**: Faculty and staff have unique insights into the suitability of study spaces for academic purposes and can act as secondary contributors to the app’s content.
- **Information to Gather**:
  - How often they recommend study spots to students
  - What kind of spaces they seek for their own work or meetings
  - Feedback on how the app could serve the academic community better

#### Campus Visitors and Future Students

- **Characteristic Types**: Unfamiliar with the campus, interested in campus facilities, may use the app to explore study environments.
- **Priority Level**: Tertiary
- **Justification**: These users can benefit from the app when visiting the campus, and their experience can influence their perception of the campus facilities.
- **Information to Gather**:
  - Ease of use and navigability of the app for first-time users
  - How the app might influence their impression of campus amenities
  - Additional features that could aid in campus exploration

---

## Personas

### Persona 1: The Quiet Seeker (Kelly)

**Characteristics**:

- Age: 21
- Major: Biomedical Engineering
- Study Preference: Quiet and isolated spaces

**Needs**:

- A quiet environment to focus on studies
- Access to power outlets for devices

**Challenges**:

- Finding consistently quiet spaces on a busy campus
- Knowing if a space will be quiet before arriving
- Whether the trip to designated spaces is safe or not

**Requirements**:

- Real-time noise level indicators
- Safe route travel

**Additional Requirements**:

- The feature to see historical data on noise levels to predict quieter times
- A review system focused on the quietness of locations

**Key Interactions/System Interfaces**:

- Filters for noise levels in the search feature
- Information page for each study spot with details on amenities like power outlets

### Persona 2: The Group Collaborator (Carlos)

**Characteristics**:

- Age: 20
- Major: Business Administration
- Study Preference: Group-friendly spaces

**Needs**:

- Large tables and spaces that allow for group discussion
- Ability to reserve study rooms

**Challenges**:

- Finding available group spaces during peak hours
- Coordinating with group members on study location

**Requirements**:

- A reservation system for study rooms
- Group-friendly location indicators

**Additional Requirements**:

- Integration with calendar apps for scheduling study sessions
- Group account features to share and save favorite locations collectively

**Key Interactions/System Interfaces**:

- Search filter for group accommodations
- Room reservation interface within the app

### Persona 3: The Convenience Seeker (Sarah)

**Characteristics**:

- Age: 19
- Major: English Literature
- Study Preference: Study spots close to the dorm with nearby food options

**Needs**:

- Easy to reach study locations
- Quick access to food and drinks

**Challenges**:

- Identifying the closest study spots to her current location
- Finding study spots with food options available

**Requirements**:

- Location-based services to find the nearest study places
- Information on study spots with nearby food options

**Additional Requirements**:

- Notifications for new food options available near favored study spots
- The ability to sort or filter search results based on proximity

**Key Interactions/System Interfaces**:

- Proximity-based search results
- Information pages for each study spot with details on nearby food options

### Persona 5: The Tech-Savvy Innovator (Ling)

**Characteristics**:

- Age: 23
- Major: Computer Science
- Study Preference: Technologically equipped spaces with high-speed Wi-Fi

**Needs**:

- Reliable and fast internet connection
- Access to technological tools and resources

**Challenges**:

- Verifying tech amenities before visiting a study spot
- Sharing information about newly discovered tech-friendly places

**Requirements**:

- Detailed listings of technological amenities at each study spot
- Ability to submit information about new study spots

**Additional Requirements**:

- Feature to rate the quality of Wi-Fi and other tech resources
- Community features to discuss tech-related study spot queries

**Key Interactions/System Interfaces**:

- Tech amenities filters in the search feature
- User submission interface for adding new study spots to the app

---

## Sketches

![[thoughts/university/twenty-three-twenty-four/hci-4hc3/Shape.svg]]

### Interactions:

- The sketch shows a navigation element at the top, potentially for a map view.
- Below the map, there are search filter options to narrow down study locations.
- Each study place listing has a checkbox feature, likely for selecting preferences or amenities.
- The "Go" button suggests an action to navigate to or learn more about the selected study place.
- A "Load more" button at the bottom indicates infinite scrolling or pagination.

### Information:

- Thebox "location" implies that users can specify current location.
- The "library" input box could be a dropdown or an auto-complete input for selecting specific libraries.
- Study places are listed with names (e.g., "lib A," "lib B") and have tags like "recommended."
- Checkboxes next to "lorem" and "ipsum" could denote available features or user preferences for study places.

### Structures:

- The structure is vertical and linear, designed for scrolling on mobile devices.
- There is a clear division between the map/search area and the list of study places.
- Interface elements are organized in blocks, which aids in understanding the hierarchy of information.

### Layout:

- The layout is straightforward, with a focus on clarity and ease of use.
- The design provides a clear path from search to selection to action, with "Go" buttons for each place.

### Potential Problems in the Design:

- **Map Interaction**: The map at the top can be unclear—does it update based on the location input, and can users interact with it?
- **Information Density**: The listings may become cluttered if there are many features or amenities associated with each place.
- **Filter Complexity**: It’s not clear how detailed the filters are. For instance, will users be able to filter by noise level, crowdedness, or proximity to food?
- **Scalability**: The "Load more" button suggests there may be many study places, but too many listings could overwhelm the user.
- **Feedback Mechanism**: There is no clear way for users to provide feedback on study places, which is crucial for a review-based app.
- **Recommendation Logic**: There's a tag for "recommended," but it's unclear what criteria are used for recommendations. This could be recommended based on users needs, feedback?
  ![[thoughts/university/twenty-three-twenty-four/hci-4hc3/Shape 6M.svg]]

### Interactions:

- The top section seems to represent a map interface, which could be interactive, allowing users to pan or zoom in/out.
- There is a search bar that implies users can search for specific locations or types of study spots.
- Below the search bar, there are saved location entries which users can presumably tap to quickly navigate to saved spots.

### Information:

- The map's purpose is likely to show the spatial arrangement of study spots relative to the user's current location.
- The search bar allows for direct input to find specific study areas.
- The "saved locations" section provides quick access to user-preferred spots.

### Structures:

- The layout is divided into three main sections: map, search, and saved locations, indicating a clear hierarchy.
- The saved locations are listed in a way that suggests they could be part of a scrollable list.

### Layout:

- The layout is simple and minimalistic, which can aid in user focus and reduce cognitive load.
- The map is given prominence, which suggests that geographical context is important in this app's usage.

### Potential Problems in the Design:

- **Map Clarity**: It is unclear if the map is static or interactive, and how much detail it provides. For instance, does it show all study spots or just the ones related to the search results?
- **Search Bar Functionality**: There is no clear indication of whether the search bar supports auto-complete or filters.
- **Visual Feedback**: It's not evident what visual feedback (like highlighting or a popup) occurs when a saved location is selected.
- **Scalability**: With only three saved spots shown, it's unclear how the app would handle a larger number of saved locations.
- **Navigation Between Screens**: There are no visible controls for navigating back to a previous screen or to a detailed view of a selected location.
- **Interaction with Saved Locations**: It's not specified what actions can be taken with saved locations—can they be edited, reordered, or deleted?

---

## Prototype

[Figma](https://www.figma.com/proto/Y1OEODbxykwOFf3vXEKP1R/Campus-Review?page-id=51%3A14320&type=design&node-id=51-14347&viewport=3542%2C672%2C0.6&t=BUHiCbyijdJ3TbOn-1&scaling=scale-down&starting-point-node-id=51%3A14346&show-proto-sidebar=1&mode=design)
