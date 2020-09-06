"""
Selecting the vertices (v) and faces (f) from face's .obj file in order
to correct it
"""

if __name__ == '__main__':
    newlines = []
    path = "../data/Wikihuman_project/obj/Emily_2_1.obj"
    with open(path, 'r') as file:
        stop = False
        while not stop:
            line = file.readline()
            if line[0:2] == "v " or line[0:2] == "f ":
                newlines.append(line)
            if line[0:8] == "# object" and line[9:19] != "Emily_head":
                stop = True

    with open("new_mesh.obj", 'w') as file:
        file.writelines(newlines)
