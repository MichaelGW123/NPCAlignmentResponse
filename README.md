# NPCAlignmentResponse
NLP Program incorporating multiple classifiers to do sentiment analysis on text input to give an NPC's response of agreement or diagreement based on the alignment of the NPC.
Primarily will be focusing on the standard roleplaying tabletop game Dungeons & Dragons (D&D) alignment chart.
The project will exclude the neutrals since Good vs Evil or Lawful vs Chaotic is already subjective, to more easily avoid confusion based on context. 
Good-Evil Classifier has the different algorithms availible to switch for ease of testing and parameter optimizatio to choose the best classifier. Same for Lawful-Chaotic Classifier.
Currently Extreme Classifier can be run and will continue to check alignment based on the classifiers training on the datasets until "Done" is input. It is currently only using the SVM model which showed highest overall results in many cases, but can be changed to any of the other models.
