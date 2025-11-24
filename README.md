# Padayani Eye Tracking
## Background
- The Exhibit is going to be kept atleast 3 metres away from the audience
- Camera is prone to glaze from nearby light
- Camera could be kept on the top or at eye level
- It is going to be kept in a lobby with a huge crowd
- Might need to take account for the body points for the eye tracker as the person is going to be a bit far away from the camera, we'd also take face points into account as we'd need to take if the person is looking at the exhibit or not

## Requirements
- Need to track a person 
- Needs to track the person who has been in the frame for the longest or the person with the closest proximity, also make sure to track the person who's looking at the installation.

## Logic for eyes

### Logic for all three pairs to focus on one person
- find the person looking at the installation, make sure their face is facing the installation
- lock into that person until, they move out of the frame
- move all the three eyes facing the person on focus
- move based on the person's position
- once out of focus, find th next person in crowd

### Logic for one pair of eyes to track one person (3 pairs, 3 persons)
- Lock the person based on First Come First Serve basis, when only one person present, fall back to the previous logic
- when two persons are looking at the installation, use a pair of eyes to focus on the person 2
- when three persons are looking at the installation, assign the third pair of eyes to focus on person 3
- when a person leaves out of focus, [make a decision on what to do]

| Detected Count | Eye Pair 1 | Eye Pair 2 | Eye Pair 3 |
| -------------- | ---------- | ---------- | ---------- |
| 1 person       | Person 1   | Person 1   | Person 1   |
| 2 persons      | Person 1   | Person 2   | Person 2   |
| 3 persons      | Person 1   | Person 2   | Person 3   |

### Logic when a person goes out of focus
- If the next person isnt found it goes back to the normal resting position

### References
- https://bikerglen.com/blog/tracking-people-with-the-googly-eyes-and-opencv/


#### Misc Inferences
- Swapping issues with people
- How to manage crowd flow as most of the crowd would be moving
- Eyes going back to default v/s Eyes tracking the next person to be tracked when the person on focus leaves
- Identify who is being tracked
- 