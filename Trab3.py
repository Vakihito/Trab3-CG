# xvfb-run python3
import glfw
from OpenGL.GL import *
import OpenGL.GL.shaders
import numpy as np
import glm
import math
import random
from PIL import Image


glfw.init()
glfw.window_hint(glfw.VISIBLE, glfw.FALSE);
altura = 1300
largura = 1300
window = glfw.create_window(largura, altura, "Malhas e Texturas", None, None)
glfw.make_context_current(window)


vertex_code = """
        attribute vec3 position;
        attribute vec2 texture_coord;
        varying vec2 out_texture;
                
        uniform mat4 model;
        uniform mat4 view;
        uniform mat4 projection;        
        
        void main(){
            gl_Position = projection * view * model * vec4(position,1.0);
            out_texture = vec2(texture_coord);
        }
        """

fragment_code = """
        uniform vec4 color;
        varying vec2 out_texture;
        uniform sampler2D samplerTexture;
        
        void main(){
            vec4 texture = texture2D(samplerTexture, out_texture);
            gl_FragColor = texture;
        }
        """

# Request a program and shader slots from GPU
program  = glCreateProgram()
vertex   = glCreateShader(GL_VERTEX_SHADER)
fragment = glCreateShader(GL_FRAGMENT_SHADER)


# Set shaders source
glShaderSource(vertex, vertex_code)
glShaderSource(fragment, fragment_code)


glCompileShader(vertex)
if not glGetShaderiv(vertex, GL_COMPILE_STATUS):
    error = glGetShaderInfoLog(vertex).decode()
    print(error)
    raise RuntimeError("Erro de compilacao do Vertex Shader")

glCompileShader(fragment)
if not glGetShaderiv(fragment, GL_COMPILE_STATUS):
    error = glGetShaderInfoLog(fragment).decode()
    print(error)
    raise RuntimeError("Erro de compilacao do Fragment Shader")


# Attach shader objects to the program
glAttachShader(program, vertex)
glAttachShader(program, fragment)


# Build program
glLinkProgram(program)
if not glGetProgramiv(program, GL_LINK_STATUS):
    print(glGetProgramInfoLog(program))
    raise RuntimeError('Linking error')
    
# Make program the default program
glUseProgram(program)




def load_model_from_file(filename):
    """Loads a Wavefront OBJ file. """
    objects = {}
    vertices = []
    texture_coords = []
    faces = []

    material = None

    # abre o arquivo obj para leitura
    for line in open(filename, "r"): ## para cada linha do arquivo .obj
        if line.startswith('#'): continue ## ignora comentarios
        values = line.split() # quebra a linha por espaço
        if not values: continue


        ### recuperando vertices
        if values[0] == 'v':
            vertices.append(values[1:4])


        ### recuperando coordenadas de textura
        elif values[0] == 'vt':
            texture_coords.append(values[1:3])

        ### recuperando faces 
        elif values[0] in ('usemtl', 'usemat'):
            material = values[1]
        elif values[0] == 'f':
            face = []
            face_texture = []
            for v in values[1:]:
                w = v.split('/')
                face.append(int(w[0]))
                if len(w) >= 2 and len(w[1]) > 0:
                    face_texture.append(int(w[1]))
                else:
                    face_texture.append(0)

            faces.append((face, face_texture, material))

    model = {}
    model['vertices'] = vertices
    model['texture'] = texture_coords
    model['faces'] = faces

    return model

glEnable(GL_TEXTURE_2D)
qtd_texturas = 50
textures = glGenTextures(qtd_texturas)
loc_color = glGetUniformLocation(program, "color")



def load_texture_from_file(texture_id, img_textura):
    glBindTexture(GL_TEXTURE_2D, texture_id)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
    img = Image.open(img_textura)
    img_width = img.size[0]
    img_height = img.size[1]
    image_data = img.tobytes("raw", "RGB", 0, -1)
    #image_data = np.array(list(img.getdata()), np.uint8)
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, img_width, img_height, 0, GL_RGB, GL_UNSIGNED_BYTE, image_data)

vertices_list = []    
textures_coord_list = []

vertices_dict = {}
counterModels = 0;
def processObjects(dir, objFile, imageFile):
    global vertices_dict, counterModels;
    print("####################################\n")

    modelo = load_model_from_file(dir + '/' + objFile)
    ### inserindo vertices do modelo no vetor de vertices
    ini_list_ver = len(vertices_list);
    print('Processando modelo |' + objFile + '| Vertice inicial:',len(vertices_list))
    for face in modelo['faces']:
        for vertice_id in face[0]:
            vertices_list.append( modelo['vertices'][vertice_id-1] )
        for texture_id in face[1]:
            textures_coord_list.append( modelo['texture'][texture_id-1] )
    end_list_ver = len(vertices_list)
    print('Processando modelo |' + objFile + '| Vertice final:',len(vertices_list))
    # adicionando ao dicionário de vertices de inicio e fim
    vertices_dict[dir] = (ini_list_ver, end_list_ver, counterModels)
    ### inserindo coordenadas de textura do modelo no vetor de texturas
    ### carregando textura equivalente e definindo um id (buffer): use um id por textura!
    load_texture_from_file(counterModels,dir + '/' +imageFile)
    counterModels += 1;

    print("####################################\n")

    

def processObjects2Textures(dir, objFile, imageFile1, imageFile2):
    global vertices_dict, counterModels 

    modelo = load_model_from_file(dir + '/' + objFile)

    ### inserindo vertices do modelo no vetor de vertices
    print("####################################\n")
    print('Processando modelo ' + objFile + 'Vertice inicial:',len(vertices_list))
    faces_visited = []
    for face in modelo['faces']:
        if face[2] not in faces_visited:
            print(face[2],' vertice inicial =',len(vertices_list))
            faces_visited.append(face[2])
        for vertice_id in face[0]:
            vertices_list.append( modelo['vertices'][vertice_id-1] )
        for texture_id in face[1]:
            textures_coord_list.append( modelo['texture'][texture_id-1] )
    print('Processando modelo '+ objFile +' Vertice final:',len(vertices_list))

    ### inserindo coordenadas de textura do modelo no vetor de texturas


    ### carregando textura equivalente e definindo um id (buffer): use um id por textura!
    load_texture_from_file(counterModels, dir + '/' + imageFile1)
    print( dir +" texture 1:", counterModels)
    counterModels += 1;

    load_texture_from_file(counterModels,dir + '/' + imageFile2)
    print(dir + " texture 2:",counterModels)
    
    vertices_dict[dir] = (counterModels - 1,counterModels)
    counterModels += 1;

    print("####################################\n")



# interior da casa
################################################################


processObjects("aya", "aya.obj", "aya.jpg")
processObjects("beagle", "beagle.obj", "beagle.jpg")
processObjects2Textures("chair", "chair.obj", "chair1.jpg", "chair2.PNG")
processObjects("cottage", "cottage.obj", "cottage2.png")
processObjects("table1", "table1.obj", "table1.png")
processObjects("floor", "floor.obj", "floor.jpg")
processObjects2Textures("sofa", "sofa.obj", "white.PNG", "wood.jpg")
processObjects("stool", "stool.obj", "stool.png")
processObjects("plant", "plant.obj", "plant.jpg")
processObjects("tv", "tv.obj", "tv.png")
processObjects("cabinet", "cabinet.obj", "cabinet.jpg")
processObjects("bed", "bed1.obj", "Texture.png")
processObjects("chair2", "chair2.obj", "chair2.jpg")
processObjects2Textures("lamp", "lamp.obj", "lamp.jpg", "luz.png")







################################################################

# exterior da casa
################################################################

processObjects("grass", "terreno2.obj", "grass.jpeg")
processObjects("street", "terreno2.obj", "street.jpg")
processObjects("water", "water.obj", "water5.jpg")
processObjects("car", "car.obj", "car.jpg")
processObjects("dogh", "doghouse.obj", "2_BaseColor.jpg")
processObjects("ball", "ball.obj", "ball.jpg")
processObjects("doberman", "dog2.obj", "Doberman_Pinscher_dif.jpg")
processObjects("cat", "cat.obj", "Cat_bump.jpg")
processObjects("sky", "terreno2.obj", "sky2.jpg")
processObjects("cobleStone", "floor.obj", "cobleStone.jpg")
processObjects("dolphin", "dolphin1.obj", "dolphin.jpg")
processObjects("whale", "whale.obj", "10054_Whale_Diffuse_v2.jpg")
processObjects("plane1", "plane1.obj", "plane1.jpg")
processObjects("plane2", "plane2.obj", "plane2.jpg")
processObjects("container", "container.obj", "12281_Container_diffuse.jpg")
processObjects("librarian", "librarian1.obj", "act_bibliotekar.jpg")
processObjects("sun", "sun2.obj", "sun.jpg")














################################################################



# Request a buffer slot from GPU
buffer = glGenBuffers(2)
vertices = np.zeros(len(vertices_list), [("position", np.float32, 3)])
vertices['position'] = vertices_list


# Upload data
glBindBuffer(GL_ARRAY_BUFFER, buffer[0])
glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_STATIC_DRAW)
stride = vertices.strides[0]
offset = ctypes.c_void_p(0)
loc_vertices = glGetAttribLocation(program, "position")
glEnableVertexAttribArray(loc_vertices)
glVertexAttribPointer(loc_vertices, 3, GL_FLOAT, False, stride, offset)
textures = np.zeros(len(textures_coord_list), [("position", np.float32, 2)]) # duas coordenadas
textures['position'] = textures_coord_list


# Upload data
glBindBuffer(GL_ARRAY_BUFFER, buffer[1])
glBufferData(GL_ARRAY_BUFFER, textures.nbytes, textures, GL_STATIC_DRAW)
stride = textures.strides[0]
offset = ctypes.c_void_p(0)
loc_texture_coord = glGetAttribLocation(program, "texture_coord")
glEnableVertexAttribArray(loc_texture_coord)
glVertexAttribPointer(loc_texture_coord, 2, GL_FLOAT, False, stride, offset)



def desenha(angle=0.0, 
            r_x=0.0, r_y=0.0, r_z=1.0,
            t_x=0.0, t_y=0.0, t_z=0.0,
            s_x=1.0, s_y=1.0, s_z=1.0,
            modelDir=""):
    mat_model = model(angle, r_x, r_y, r_z, t_x, t_y, t_z, s_x, s_y, s_z)
    loc_model = glGetUniformLocation(program, "model")
    glUniformMatrix4fv(loc_model, 1, GL_TRUE, mat_model)
    #define id da textura do modelo
    glBindTexture(GL_TEXTURE_2D, vertices_dict[modelDir][2])
    # desenha o modelo
    glDrawArrays(GL_TRIANGLES, vertices_dict[modelDir][0], vertices_dict[modelDir][1] - vertices_dict[modelDir][0]) ## renderizando

def desenhaM2(angle=0.0, 
            r_x=0.0, r_y=0.0, r_z=1.0,
            t_x=0.0, t_y=0.0, t_z=0.0,
            s_x=1.0, s_y=1.0, s_z=1.0,
            modelDir=""):
    mat_model = model2(angle, r_x, r_y, r_z, t_x, t_y, t_z, s_x, s_y, s_z)
    loc_model = glGetUniformLocation(program, "model")
    glUniformMatrix4fv(loc_model, 1, GL_TRUE, mat_model)
    #define id da textura do modelo
    glBindTexture(GL_TEXTURE_2D, vertices_dict[modelDir][2])
    # desenha o modelo
    glDrawArrays(GL_TRIANGLES, vertices_dict[modelDir][0], vertices_dict[modelDir][1] - vertices_dict[modelDir][0]) ## renderizando





def desenha_chair(angle=0.0, 
            r_x=0.0, r_y=0.0, r_z=1.0,
            t_x=0.0, t_y=0.0, t_z=0.0,
            s_x=1.0, s_y=1.0, s_z=1.0,
            modelDir=""):
    mat_model = model(angle, r_x, r_y, r_z, t_x, t_y, t_z, s_x, s_y, s_z)
    loc_model = glGetUniformLocation(program, "model")
    glUniformMatrix4fv(loc_model, 1, GL_TRUE, mat_model)
    
    
    #define id da textura do modelo
    glBindTexture(GL_TEXTURE_2D, vertices_dict[modelDir][1])
    glDrawArrays(GL_TRIANGLES, 197592, 203592-197592) ## renderizando
    
    #define id da textura do modelo
    glBindTexture(GL_TEXTURE_2D, vertices_dict[modelDir][1])
    glDrawArrays(GL_TRIANGLES, 251304, 499884-251304) ## renderizando

    # define id da textura do modelo
    glBindTexture(GL_TEXTURE_2D, vertices_dict[modelDir][0])
    glDrawArrays(GL_TRIANGLES, 203598, 227358-203598) ## renderizando
    # define id da textura do modelo
    glBindTexture(GL_TEXTURE_2D, vertices_dict[modelDir][0])
    glDrawArrays(GL_TRIANGLES, 227358, 251304-227358) ## renderizando
    
def desenha_cottage(angle=0.0, 
            r_x=0.0, r_y=0.0, r_z=1.0,
            t_x=0.0, t_y=0.0, t_z=0.0,
            s_x=1.0, s_y=1.0, s_z=1.0,
            modelDir=""):
    mat_model = model(angle, r_x, r_y, r_z, t_x, t_y, t_z, s_x, s_y, s_z)
    loc_model = glGetUniformLocation(program, "model")
    glUniformMatrix4fv(loc_model, 1, GL_TRUE, mat_model)
    
    
    #define id da textura do modelo
    glBindTexture(GL_TEXTURE_2D, vertices_dict[modelDir][1])
    glDrawArrays(GL_TRIANGLES, 499878, 512721-499878) ## renderizando
        
def desenha_sofa(angle=0.0, 
            r_x=0.0, r_y=0.0, r_z=1.0,
            t_x=0.0, t_y=0.0, t_z=0.0,
            s_x=1.0, s_y=1.0, s_z=1.0,
            modelDir="sofa"):
    mat_model = model(angle, r_x, r_y, r_z, t_x, t_y, t_z, s_x, s_y, s_z)
    loc_model = glGetUniformLocation(program, "model")
    glUniformMatrix4fv(loc_model, 1, GL_TRUE, mat_model)
    
    
    #define id da textura do modelo
    glBindTexture(GL_TEXTURE_2D, vertices_dict[modelDir][0])
    c = 87172 * 3
    
    glDrawArrays(GL_TRIANGLES, 535947, c) ## renderizando
    c = 535947 + c
    #define id da textura do modelo
    glBindTexture(GL_TEXTURE_2D, vertices_dict[modelDir][1])
    glDrawArrays(GL_TRIANGLES, c, 837771 - c) ## renderizando

def desenha_lamp(angle=0.0, 
            r_x=0.0, r_y=0.0, r_z=1.0,
            t_x=0.0, t_y=0.0, t_z=0.0,
            s_x=1.0, s_y=1.0, s_z=1.0,
            modelDir="lamp"):
    mat_model = model(angle, r_x, r_y, r_z, t_x, t_y, t_z, s_x, s_y, s_z)
    loc_model = glGetUniformLocation(program, "model")
    glUniformMatrix4fv(loc_model, 1, GL_TRUE, mat_model)
    
    #define id da textura do modelo
    glBindTexture(GL_TEXTURE_2D, vertices_dict[modelDir][0])
    glDrawArrays(GL_TRIANGLES, 1745787, 1830663 - 1745787) ## renderizando

    #define id da textura do modelo
    glBindTexture(GL_TEXTURE_2D, vertices_dict[modelDir][1])
    glDrawArrays(GL_TRIANGLES, 1830663, 1865640 - 1830663) ## renderizando

    
    #define id da textura do modelo
    glBindTexture(GL_TEXTURE_2D, vertices_dict[modelDir][0])
    glDrawArrays(GL_TRIANGLES, 1865640, 1890780 - 1865640) ## renderizando



cameraPos   = glm.vec3(0.0,  10.0,  1.0);
cameraFront = glm.vec3(0.0,  0.0, -1.0);
cameraUp    = glm.vec3(0.0,  1.0,  0.0);


polygonal_mode = False

def key_event(window,key,scancode,action,mods):
    global cameraPos, cameraFront, cameraUp, polygonal_mode
    
    cameraSpeed = 0.5
    if key == 87 and (action==1 or action==2): # tecla W
        cameraPos += cameraSpeed * cameraFront
    
    if key == 83 and (action==1 or action==2): # tecla S
        cameraPos -= cameraSpeed * cameraFront


    if cameraPos[1] > 89.8:
        cameraPos = glm.vec3((cameraPos[0], 89.8, cameraPos[2]))
    elif cameraPos[1] < -1.8:
        cameraPos = glm.vec3((cameraPos[0], -1.8, cameraPos[2]))
    
    if key == 65 and (action==1 or action==2): # tecla A
        cameraPos -= glm.normalize(glm.cross(cameraFront, cameraUp)) * cameraSpeed
        
    if key == 68 and (action==1 or action==2): # tecla D
        cameraPos += glm.normalize(glm.cross(cameraFront, cameraUp)) * cameraSpeed
        
    if key == 80 and action==1 and polygonal_mode==True:
        polygonal_mode=False
    else:
        if key == 80 and action==1 and polygonal_mode==False:
            polygonal_mode=True
        
    

firstMouse = True
yaw = -90.0 
pitch = 0.0
lastX =  largura/2
lastY =  altura/2

def mouse_event(window, xpos, ypos):
    global firstMouse, cameraFront, yaw, pitch, lastX, lastY
    if firstMouse:
        lastX = xpos
        lastY = ypos
        firstMouse = False

    xoffset = xpos - lastX
    yoffset = lastY - ypos
    lastX = xpos
    lastY = ypos

    sensitivity = 0.3 
    xoffset *= sensitivity
    yoffset *= sensitivity

    yaw += xoffset;
    pitch += yoffset;

    
    if pitch >= 90.0: pitch = 90.0
    if pitch <= -90.0: pitch = -90.0

    front = glm.vec3()
    front.x = math.cos(glm.radians(yaw)) * math.cos(glm.radians(pitch))
    front.y = math.sin(glm.radians(pitch))
    front.z = math.sin(glm.radians(yaw)) * math.cos(glm.radians(pitch))
    cameraFront = glm.normalize(front)


    
glfw.set_key_callback(window,key_event)
glfw.set_cursor_pos_callback(window, mouse_event)

def model(angle, r_x, r_y, r_z, t_x, t_y, t_z, s_x, s_y, s_z):
    
    angle = math.radians(angle)
    
    matrix_transform = glm.mat4(1.0) # instanciando uma matriz identidade
       
    # aplicando rotacao
    matrix_transform = glm.rotate(matrix_transform, angle, glm.vec3(r_x, r_y, r_z))
        
  
    # aplicando translacao
    matrix_transform = glm.translate(matrix_transform, glm.vec3(t_x, t_y, t_z))    
    
    # aplicando escala
    matrix_transform = glm.scale(matrix_transform, glm.vec3(s_x, s_y, s_z))
    
    matrix_transform = np.array(matrix_transform).T # pegando a transposta da matriz (glm trabalha com ela invertida)
    
    return matrix_transform

def model2(angle, r_x, r_y, r_z, t_x, t_y, t_z, s_x, s_y, s_z):
    
    angle = math.radians(angle)
    
    matrix_transform = glm.mat4(1.0) # instanciando uma matriz identidade
       
    # aplicando translacao
    matrix_transform = glm.translate(matrix_transform, glm.vec3(t_x, t_y, t_z))    
    
    # aplicando escala
    matrix_transform = glm.scale(matrix_transform, glm.vec3(s_x, s_y, s_z))

    # aplicando rotacao
    matrix_transform = glm.rotate(matrix_transform, angle, glm.vec3(r_x, r_y, r_z))
    
    matrix_transform = np.array(matrix_transform).T # pegando a transposta da matriz (glm trabalha com ela invertida)
    
    return matrix_transform

def view():
    global cameraPos, cameraFront, cameraUp
    mat_view = glm.lookAt(cameraPos, cameraPos + cameraFront, cameraUp);
    mat_view = np.array(mat_view)
    return mat_view

def projection():
    global altura, largura
    # perspective parameters: fovy, aspect, near, far
    mat_projection = glm.perspective(glm.radians(45.0), largura/altura, 0.1, 1000.0)
    mat_projection = np.array(mat_projection)    
    return mat_projection

def circle(angle, raio):
    return (raio * math.sin(math.radians(angle)), raio * math.cos(math.radians(angle)) )

angle_dolphin = -90
def fish_move(x, raio, ini, dol):
    global angle_dolphin
    angle = math.radians(x - ini)
    angle *= 7
    if dol :
        angle_dolphin -= 1
        if  angle_dolphin < -90 and  -4.5 >  raio * math.sin(angle)  :
            angle_dolphin = 90

    return raio * math.sin(angle)


def elipse(rx, rz, anglePlaneAux ):
    
    angleAux = math.radians( anglePlaneAux )
    xr = rx * math.sin( angleAux )
    zr = rz * math.cos( angleAux )

    return (xr, zr)




glfw.show_window(window)
glfw.set_cursor_pos(window, lastX, lastY)

glEnable(GL_DEPTH_TEST) ### importante para 3D


def ball_move():
    #gt ^ 2/2
    return - (ball_position ** 2) + 9

car_position = - 125
ball_position = -3
x_ball_postion = 0.1
angle_run_cat_dog = 0
dolphin_position = 40
angle_dolphin1 = 0
whale_position = 40
anglePlane = 0.0
sun_angle = 0.0

while not glfw.window_should_close(window):

    glfw.poll_events() 
    
    
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    
    glClearColor(1.0, 1.0, 1.0, 1.0)
    
    if polygonal_mode:
        glPolygonMode(GL_FRONT_AND_BACK,GL_LINE)
    else:
        glPolygonMode(GL_FRONT_AND_BACK,GL_FILL)
    
    

    
    # interior da casa
    


    desenha(s_x=0.005, s_y=0.005, s_z=0.005,t_x =-4, t_z=-19.8, modelDir="aya");
    desenha(s_x=20, s_y=20, s_z=20, t_z = -10,t_y= 0,modelDir="floor")
    desenha(s_x=6, s_y=6, s_z=6, t_x = -14,t_z= 13,modelDir="floor")    
    desenha_chair(s_x=0.04, s_y=0.04, s_z=0.04,t_x =-4 ,t_z=-20, modelDir="chair");
    desenha(angle=-90,r_x=1.0 ,r_z=0.0,s_x=0.05, s_y=0.05, s_z=0.05,t_y= -6, t_z=0,modelDir="beagle")
    desenha_lamp(s_x=0.05, s_y=0.05, s_z=0.05,t_y= 27, t_z=-6.5,modelDir="lamp")
    desenha(s_x=3.58, s_y=5, s_z=4,t_x= -0.7, t_y = -2,t_z= -8, modelDir="cottage")
    desenha(angle=-90, r_y=1.0 ,r_z=0.0, s_x=0.013, s_y=0.013, s_z=0.013, t_y= 3, t_x=-18, t_z=-18,modelDir="table1")
    desenha_sofa(s_x=0.05, s_y=0.05, s_z=0.05, t_x=-17, t_y = 4,t_z=-22 ,modelDir="sofa")
    desenha(s_x=2, s_y=0.5, s_z=2, t_x=-13.5, t_z = -14, modelDir="stool")
    desenha(s_x=0.4, s_y=0.4, s_z=0.4,t_x=-13.5, t_y= 2 ,t_z = -14,modelDir="plant")
    desenha(s_x=0.8, s_y=0.8, s_z=0.8,t_y= 0, t_x=18, t_z=-13,modelDir="plant")
    desenha_chair(angle=90, r_y=1.0 ,r_z=0.0,s_x=0.04, s_y=0.04, s_z=0.04,  t_x=16, t_z=16, modelDir="chair");
    desenha(s_x=0.08, s_y=0.06, s_z=0.05,t_x=-13.5, t_z = -5,modelDir="cabinet")
    desenha(angle=180, r_y=1.0,r_z=0.0, t_x=13.5, t_y= 2.3,t_z = 4, modelDir="tv")
    desenha(angle=-90 ,r_y=1.0,r_z=0.0,s_x=2.5,s_y=2.5,s_z=2.5,t_x= 5, t_z=-15, t_y= 2,modelDir="bed")
    desenha(angle=90,r_y=0.1,r_z=0.0,  s_x=0.08, s_y=0.06, s_z=0.04,t_y=2.0,t_x=-12.5, t_z = -20.0,modelDir="cabinet")
    desenha(s_x=0.4, s_y=0.4, s_z=0.4, t_y=4.3,t_x=-18, t_z= 7,modelDir="plant")
    desenha(s_x=0.4, s_y=0.4, s_z=0.4, t_y=4.3,t_x=-18, t_z= 16,modelDir="plant")
    desenha(angle=-90,r_y=0.1,r_z=0.0,  s_x=1.1, s_y=1.1, s_z=1.1,t_x=12.5,  t_z = 16,modelDir="chair2")
    desenha(angle=-90,r_y=0.1,r_z=0.0,  s_x=1.1, s_y=1.1, s_z=1.1,t_x=10,  t_z = 16,modelDir="chair2")
    

    #exterior da casa


    desenha(s_x=140, s_y=140, s_z=140, t_z=-40 , t_y = -2, modelDir="grass")
    desenha(s_x=10, s_z= 20, t_z= -50, t_y= -1.8,modelDir= "cobleStone")
    desenha(s_x=140, s_y=140, s_z=140, t_z=-40 , t_y = 90, modelDir="sky")  

    if  sun_angle > 360:
        sun_angle = 0.0
    sun_angle += 0.3;

    desenhaM2(angle =sun_angle * 3, r_y=1.0, r_z=0.0, s_x= 5.0 ,s_y=5.0 ,s_z=5.0, t_z = -70 + 100 * math.sin(math.radians(sun_angle)) + 50, t_x = 100 * math.cos(math.radians(sun_angle)) ,t_y = 80, modelDir="sun")

    desenha(angle = 90 , r_z= 0 ,r_y = 1,s_x=140.0, s_z=14.0, t_z = -40, t_x = 40 ,t_y = -1.8, modelDir="street")
    desenha(angle = 90 , r_z= 0 ,r_y = 1,s_x=140.0, s_z=20.0, t_z = 50, t_x = 40 ,t_y = -1.8, modelDir="water")
  
    # desenhando golfinhos

    if  dolphin_position < -150:
        dolphin_position = 40
    dolphin_position -= 0.19
    ydp = fish_move(dolphin_position, 5, 40, True)
    desenhaM2(angle= angle_dolphin,r_x=1.0,r_z = 0.0 ,s_x=1.1, s_y=1.1, s_z=1.1,t_z = dolphin_position , t_x = 45 ,t_y = -2.0 + ydp , modelDir="dolphin")
    desenhaM2(angle= angle_dolphin,r_x=1.0,r_z = 0.0 ,s_x=1.1, s_y=1.1, s_z=1.1,t_z = dolphin_position + 10, t_x = 40 ,t_y = -2.0 + ydp , modelDir="dolphin")

    # desenha baleia

    if  whale_position < -150:
        whale_position = 40
    whale_position -= 0.1
    ydw = fish_move(whale_position, 1, 40, False)
    desenhaM2(angle=90,r_x=0.0,r_y = 1, r_z = 0 ,s_x=0.2, s_y=0.2, s_z=0.2,t_z = whale_position , t_x = 55 ,t_y = -2.5 + ydw , modelDir="whale")

    # desenha cachorro e casa do cachorro
    desenha(s_z=4, s_y=3, s_x=2 ,t_z = -50, t_x = -15 ,t_y = -2, modelDir="dogh")
    desenha(s_x=0.06, s_y=0.04, s_z=0.06,t_y= -1.7, t_x = -14 , t_z=-50 ,modelDir="doberman")


    # desenhando os aviões

    anglePlane += 1
    if anglePlane > 360:
        anglePlane = 0.0

    anglePlane2 = anglePlane + 90
    (planex1, planey2) = elipse( 50 , 40, anglePlane)
    desenhaM2(angle=anglePlane,r_y=1.0,r_z=0,t_y= 40 + 10 * math.sin(math.radians(1.5 * anglePlane)), t_x = 5  + planex1, t_z=-70 + planey2,modelDir="plane1")
    
    (planex1, planey2) = elipse( 50 , 40, anglePlane2)
    desenhaM2(angle=anglePlane2, s_z=0.7, s_y=0.7, s_x=0.7 ,r_y=1.0,r_z=0,t_y= 40 + 8 * math.sin(math.radians(1.5 * anglePlane)), t_x = 5  + planex1, t_z=-70 + planey2,modelDir="plane2")
    

    # desenha container

    desenhaM2( s_z=0.8,s_y=0.6,s_x=0.8, t_y=4.8,t_x=5 , t_z= -100 ,modelDir="container")
    desenha_lamp(s_x=0.05, s_y=0.05, s_z=0.05,t_y=9.8,t_x=5.0 , t_z= -90.0)
    desenhaM2( s_z=0.5,s_y=0.5,s_x=0.5, t_y=-1.0,t_x=-2 , t_z= -90 ,modelDir="librarian")
    desenhaM2(angle=180,r_y=1.0,r_z=0.0, s_z=0.5,s_y=0.5,s_x=0.5, t_y=-1.0,t_x=12, t_z= -90 ,modelDir="librarian")




    # Faz a bola quicar
    if ball_position > 3:
        x_ball_postion = -0.1;
    elif ball_position < -3:
        x_ball_postion = 0.1;
    ball_position += x_ball_postion

    desenha(s_x=0.2, s_y=0.2, s_z=0.2,t_z= -55, t_x = 20 , t_y= ball_move(),modelDir="ball")

    
    # movendo o carro
    if car_position < -125:
        car_position = 40
    car_position -= 1

    desenha(s_x=3.0, s_y=3.0 ,s_z=3.0, t_z = car_position, t_x = -33 ,t_y = -1.8, modelDir="car")    





    mat_view = view()
    loc_view = glGetUniformLocation(program, "view")
    glUniformMatrix4fv(loc_view, 1, GL_FALSE, mat_view)

    mat_projection = projection()
    loc_projection = glGetUniformLocation(program, "projection")
    glUniformMatrix4fv(loc_projection, 1, GL_FALSE, mat_projection)    
    
    

    
    glfw.swap_buffers(window)

glfw.terminate()