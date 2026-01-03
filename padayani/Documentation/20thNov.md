# Padayani Eye Tracking
## Background
- The Exhibit is going to be kept atleast 3 metres away from the audience
- Camera is prone to glaze from nearby light
- Camera could be kept on the top or at eye level
- It is going to be kept in a lobby with a huge crowd
- I have a web cam (logitech) and a stereo camera (ZED 1), we can use either
- This is going to be used in an edge device (Rasp pi, etc)

## Requirements
- Need to track a person, keep it optimized but it should be fail proof
- Need to lock the person sitting on a bench, once they sit there for 2 seconds, start tracking them, and lock into them until they move out of the camera's view 

## Logic for eyes

### Logic for all three pairs to focus on one person
- find the person looking at the installation, make sure their face is facing the installation and sitting in the bench
- lock into that person until, they move out of the frame
- move all the three eyes facing the person on focus
- move based on the person's position
- once out of focus, find the next person in crowd

### Logic for one pair of eyes to track one person (3 pairs, 3 persons)
- Lock the person based on First Come First Serve basis, when only one person present, fall back to the previous logic
  
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
- Swapping issues with people when someone overlaps
- How to manage crowd flow as most of the crowd would be moving
- Eyes going back to default v/s Eyes tracking the next person to be tracked when the person on focus leaves
- Identify who is being tracked