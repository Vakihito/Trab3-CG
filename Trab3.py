import glfw
from OpenGL.GL import *
import OpenGL.GL.shaders
import numpy as np
import glm
import math
from PIL import Image

glfw.init()
glfw.window_hint(glfw.VISIBLE, glfw.FALSE);
altura = 1200
largura = 1200
window = glfw.create_window(largura, altura, "Trabalho 3", None, None)
glfw.make_context_current(window)

vertex_code = """
        attribute vec3 position;
        attribute vec2 texture_coord;
        attribute vec3 normals;
        
       
        varying vec2 out_texture;
        varying vec3 out_fragPos;
        varying vec3 out_normal;
                
        uniform mat4 model;
        uniform mat4 view;
        uniform mat4 projection;        
        
        void main(){
            gl_Position = projection * view * model * vec4(position,1.0);
            out_texture = vec2(texture_coord);
            out_fragPos = vec3(model * vec4(position, 1.0));
            out_normal = normals;            
        }
        """

fragment_code = """

        uniform vec3 viewPos; // define coordenadas com a posicao da camera/observador



        // parametros da iluminacao ambiente e difusa

        uniform vec3 lightPos1; // define coordenadas de posicao da luz #1
        uniform float ka_i; // coeficiente de reflexao ambiente interior
        uniform float kd_i; // coeficiente de reflexao difusa exterior
        uniform float ks_i; // coeficiente de reflexao especular
        uniform float ns_i; // expoente de reflexao especular
        
        




        // parametros da iluminacao especular

        uniform vec3 lightPos2; // define coordenadas de posicao da luz #2
        uniform float ka_e; // coeficiente de reflexao ambiente interior
        uniform float kd_e; // coeficiente de reflexao difusa exterior
        uniform float ks_e; // coeficiente de reflexao ambiente interior
        uniform float ns_e; // coeficiente de reflexao difusa exterior


        
        // parametro com a cor da(s) fonte(s) de iluminacao
        vec3 lightColor = vec3(1.0, 1.0, 1.0);

        // parametros recebidos do vertex shader
        varying vec2 out_texture; // recebido do vertex shader
        varying vec3 out_normal; // recebido do vertex shader
        varying vec3 out_fragPos; // recebido do vertex shader
        uniform sampler2D samplerTexture;
        
        
        
        void main(){
            ////////////////////////
            // Luz #1
            ////////////////////////
            
            // calculando reflexao ambiente interior
            vec3 ambient_i = ka_i * lightColor; 
            
            // calculando reflexao difusa
            vec3 norm1 = normalize(out_normal); // normaliza vetores perpendiculares
            vec3 lightDir1 = normalize(lightPos1 - out_fragPos); // direcao da luz
            float diff1 = max(dot(norm1, lightDir1), 0.0); // verifica limite angular (entre 0 e 90)
            vec3 diffuse1 = kd_i * diff1 * lightColor; // iluminacao difusa
            
            // calculando reflexao especular
            vec3 viewDir1 = normalize(viewPos - out_fragPos); // direcao do observador/camera
            vec3 reflectDir1 = reflect(-lightDir1, norm1); // direcao da reflexao
            float spec1 = pow(max(dot(viewDir1, reflectDir1), 0.0), ns_i);
            vec3 specular1 = ks_i * spec1 * lightColor;    
            
            
            ////////////////////////
            // Luz #2
            ////////////////////////

            // calculando reflexao ambiente interior
            vec3 ambient_e = ka_e * lightColor; 
            
            // calculando reflexao difusa
            vec3 norm2 = normalize(out_normal); // normaliza vetores perpendiculares
            vec3 lightDir2 = normalize(lightPos2 - out_fragPos); // direcao da luz
            float diff2 = max(dot(norm2, lightDir2), 0.0); // verifica limite angular (entre 0 e 90)
            vec3 diffuse2 = kd_e * diff2 * lightColor; // iluminacao difusa
            
            // calculando reflexao especular
            vec3 viewDir2 = normalize(viewPos - out_fragPos); // direcao do observador/camera
            vec3 reflectDir2 = reflect(-lightDir2, norm2); // direcao da reflexao
            float spec2 = pow(max(dot(viewDir2, reflectDir2), 0.0), ns_e);
            vec3 specular2 = ks_e * spec2 * lightColor;    
            
            ////////////////////////
            // Combinando as duas fontes
            ////////////////////////
            
            // aplicando o modelo de iluminacao
            vec4 texture = texture2D(samplerTexture, out_texture);
            vec4 result = vec4((ambient_e + ambient_i + diffuse1 + diffuse2 + specular1 + specular2),1.0) * texture; // aplica iluminacao
            gl_FragColor = result;

        }
        """

# Request a program and shader slots from GPU
program  = glCreateProgram()
vertex   = glCreateShader(GL_VERTEX_SHADER)
fragment = glCreateShader(GL_FRAGMENT_SHADER)

# Set shaders source
glShaderSource(vertex, vertex_code)
glShaderSource(fragment, fragment_code)


# Compile shaders
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
    normals = []
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

        ### recuperando vertices
        if values[0] == 'vn':
            normals.append(values[1:4])

        ### recuperando coordenadas de textura
        elif values[0] == 'vt':
            texture_coords.append(values[1:3])

        ### recuperando faces 
        elif values[0] in ('usemtl', 'usemat'):
            material = values[1]
        elif values[0] == 'f':
            face = []
            face_texture = []
            face_normals = []
            for v in values[1:]:
                w = v.split('/')
                face.append(int(w[0]))
                face_normals.append(int(w[2]))
                if len(w) >= 2 and len(w[1]) > 0:
                    face_texture.append(int(w[1]))
                else:
                    face_texture.append(0)

            faces.append((face, face_texture, face_normals, material))

    model = {}
    model['vertices'] = vertices
    model['texture'] = texture_coords
    model['faces'] = faces
    model['normals'] = normals

    return model

glEnable(GL_TEXTURE_2D)
qtd_texturas = 70
textures = glGenTextures(qtd_texturas)

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
normals_list = []    
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
        for normal_id in face[2]:
            normals_list.append( modelo['normals'][normal_id-1] )
    end_list_ver = len(vertices_list)
    print('Processando modelo |' + objFile + '| Vertice final:',len(vertices_list))
    # adicionando ao dicionário de vertices de inicio e fim
    vertices_dict[dir] = (ini_list_ver, end_list_ver, counterModels)
    ### inserindo coordenadas de textura do modelo no vetor de texturas
    ### carregando textura equivalente e definindo um id (buffer): use um id por textura!
    load_texture_from_file(counterModels,dir + '/' +imageFile)
    counterModels += 1;

    print("####################################\n")

def processObjects_no_normal(dir, objFile, imageFile):
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
        for vertice_id in face[0]:
            vertices_list.append( modelo['vertices'][vertice_id-1] )
        for texture_id in face[1]:
            textures_coord_list.append( modelo['texture'][texture_id-1] )
        for normal_id in face[2]:
            normals_list.append( modelo['normals'][normal_id-1] )
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


# processObjects("aya", "aya.obj", "aya.jpg")
# processObjects("beagle", "beagle.obj", "beagle.jpg")
# processObjects2Textures("chair", "chair.obj", "chair1.jpg", "chair2.PNG")
# processObjects("cottage", "cottage.obj", "cottage2.png")
# processObjects("table1", "table1.obj", "table1.png")
# processObjects("floor1", "floor.obj", "floor.jpg")
# processObjects2Textures("sofa", "sofa.obj", "white.PNG", "wood.jpg")
# processObjects("stool", "stool.obj", "stool.png")
# processObjects("plant", "plant.obj", "plant.jpg")
# processObjects("tv", "tv.obj", "tv.png")
# processObjects("cabinet", "cabinet.obj", "cabinet.jpg")
# processObjects("bed1", "bed.obj", "bed.jpg")
# processObjects("chair2", "chair2.obj", "chair2.jpg")
# processObjects("luz", "caixa2.obj", "luz.png")
# processObjects("caixa", "caixa2.obj", "wood.jpg")
# processObjects("floor", "floor2.obj", "floor.jpg")








################################################################

# exterior da casa
################################################################

# processObjects("grass", "terreno2.obj", "grass.jpeg")
# processObjects("street", "terreno2.obj", "street.jpg")
# processObjects("water", "water.obj", "water5.jpg")
# processObjects("car", "car.obj", "car.jpg")
# processObjects("dogh", "doghouse.obj", "2_BaseColor.jpg")
# processObjects("ball", "ball.obj", "ball.jpg")
# processObjects("doberman", "dog2.obj", "Doberman_Pinscher_dif.jpg")
# processObjects("cat", "cat.obj", "Cat_bump.jpg")
# processObjects("sky", "terreno.obj", "sky2.jpg")
# processObjects("cobleStone", "floor.obj", "cobleStone.jpg")
# processObjects("dolphin", "dolphin1.obj", "dolphin.jpg")
# processObjects("whale", "whale2.obj", "10054_Whale_Diffuse_v2.jpg")
# processObjects("plane1", "plane1.obj", "plane1.jpg")
# processObjects("plane2", "plane2.obj", "plane2.jpg")
# processObjects("container", "container.obj", "12281_Container_diffuse.jpg")
processObjects("librarian", "librarian1.obj", "act_bibliotekar.jpg")
processObjects("monstro", "monstro.obj", "monstro.jpg")
processObjects("caixa2", "caixa.obj", "caixa.jpg")
processObjects("sun", "moon.obj", "moon.jpg")



################################################################




# Request a buffer slot from GPU
buffer = glGenBuffers(3)

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

normals = np.zeros(len(normals_list), [("position", np.float32, 3)]) # três coordenadas
normals['position'] = normals_list


# Upload coordenadas normals de cada vertice
glBindBuffer(GL_ARRAY_BUFFER, buffer[2])
glBufferData(GL_ARRAY_BUFFER, normals.nbytes, normals, GL_STATIC_DRAW)
stride = normals.strides[0]
offset = ctypes.c_void_p(0)
loc_normals_coord = glGetAttribLocation(program, "normals")
glEnableVertexAttribArray(loc_normals_coord)
glVertexAttribPointer(loc_normals_coord, 3, GL_FLOAT, False, stride, offset)


def desenha_lamp(angle=0.0, 
            r_x=0.0, r_y=0.0, r_z=1.0,
            t_x=0.0, t_y=0.0, t_z=0.0,
            s_x=1.0, s_y=1.0, s_z=1.0,
            modelDir="luz"):
    mat_model = model2(angle, r_x, r_y, r_z, t_x, t_y, t_z, s_x, s_y, s_z)
    loc_model = glGetUniformLocation(program, "model")
    glUniformMatrix4fv(loc_model, 1, GL_TRUE, mat_model)

    #### define parametros de ilumincao do modelo
    ka = 1 # coeficiente de reflexao ambiente do modelo
    kd = 1 # coeficiente de reflexao difusa do modelo
    ks = 1 # coeficiente de reflexao especular do modelo
    ns = 1000.0 # expoente de reflexao especular
    
    loc_ka = glGetUniformLocation(program, "ka_i") # recuperando localizacao da variavel ka na GPU
    glUniform1f(loc_ka, ka) ### envia ka pra gpu
    
    loc_kd = glGetUniformLocation(program, "kd_i") # recuperando localizacao da variavel kd na GPU
    glUniform1f(loc_kd, kd) ### envia kd pra gpu    
    
    loc_ks = glGetUniformLocation(program, "ks_i") # recuperando localizacao da variavel ks na GPU
    glUniform1f(loc_ks, ks) ### envia ns pra gpu        
    
    loc_ns = glGetUniformLocation(program, "ns_i") # recuperando localizacao da variavel ns na GPU
    glUniform1f(loc_ns, ns) ### envia ns pra gpu            
    
    loc_light_pos = glGetUniformLocation(program, "lightPos1") # recuperando localizacao da variavel lightPos na GPU
    glUniform3f(loc_light_pos, t_x, t_y, t_z) ### posicao da fonte de luz

    glBindTexture(GL_TEXTURE_2D, vertices_dict[modelDir][2])
    # desenha o modelo
    glDrawArrays(GL_TRIANGLES, vertices_dict[modelDir][0], vertices_dict[modelDir][1] - vertices_dict[modelDir][0]) ## renderizando
    
    

def desenha_sun(angle=0.0, 
            r_x=0.0, r_y=0.0, r_z=1.0,
            t_x=0.0, t_y=0.0, t_z=0.0,
            s_x=1.0, s_y=1.0, s_z=1.0,
            modelDir="sun"):
        
    mat_model = model2(angle, r_x, r_y, r_z, t_x, t_y, t_z, s_x, s_y, s_z)
    loc_model = glGetUniformLocation(program, "model")
    glUniformMatrix4fv(loc_model, 1, GL_TRUE, mat_model)
       
    
    #### define parametros de ilumincao do modelo
    ka = 1 # coeficiente de reflexao ambiente do modelo
    kd = 1 # coeficiente de reflexao difusa do modelo
    ks = 1 # coeficiente de reflexao especular do modelo
    ns = 1000.0 # expoente de reflexao especular
    
    loc_ka = glGetUniformLocation(program, "ka_e") # recuperando localizacao da variavel ka na GPU
    glUniform1f(loc_ka, ka) ### envia ka pra gpu
    
    loc_kd = glGetUniformLocation(program, "kd_e") # recuperando localizacao da variavel kd na GPU
    glUniform1f(loc_kd, kd) ### envia kd pra gpu    
    
    loc_ks = glGetUniformLocation(program, "ks_e") # recuperando localizacao da variavel ks na GPU
    glUniform1f(loc_ks, ks) ### envia ns pra gpu        
    
    loc_ns = glGetUniformLocation(program, "ns_e") # recuperando localizacao da variavel ns na GPU
    glUniform1f(loc_ns, ns) ### envia ns pra gpu            
    
    loc_light_pos = glGetUniformLocation(program, "lightPos2") # recuperando localizacao da variavel lightPos na GPU
    glUniform3f(loc_light_pos, t_x, t_y, t_z) ### posicao da fonte de luz
        
    
    #define id da textura do modelo
    glBindTexture(GL_TEXTURE_2D, vertices_dict[modelDir][2])
    
    
    # desenha o modelo
    glDrawArrays(GL_TRIANGLES, vertices_dict[modelDir][0], vertices_dict[modelDir][1] - vertices_dict[modelDir][0]) ## renderizando


def sum_k(ka ,kd ,ks):
    ka_aux = ka 
    kd_aux = kd
    ks_aux = ks

    ka_aux += ka_add
    kd_aux += kd_add
    ks_aux += ks_add

    if ka_aux + ka_add > 1:
        ka_aux = 1

    if kd_aux + kd_add > 1:
        kd_aux = 1

    if ks_aux + ks_add > 1:
        ks_aux = 1
    
    if ka_aux + ka_add < 0:
        ka_aux = 0
        
    if kd_aux + kd_add < 0:
        kd_aux = 0
    
    if ks_aux + ks_add < 0:
        ks_aux = 0
    return (ka_aux, kd_aux, ks_aux)

def set_zero_e():
    loc_ka = glGetUniformLocation(program, "ka_e") # recuperando localizacao da variavel ka na GPU
    glUniform1f(loc_ka, 0.01) ### envia ka pra gpu
    
    loc_kd = glGetUniformLocation(program, "kd_e") # recuperando localizacao da variavel kd na GPU
    glUniform1f(loc_kd, 0.0) ### envia kd pra gpu    

    loc_ks = glGetUniformLocation(program, "ks_e") # recuperando localizacao da variavel ks na GPU
    glUniform1f(loc_ks, 0.0) ### envia ks pra gpu        
    
    loc_ns = glGetUniformLocation(program, "ns_e") # recuperando localizacao da variavel ns na GPU
    glUniform1f(loc_ns, 1.0) ### envia ns pra gpu        

def set_zero_i():
    loc_ka = glGetUniformLocation(program, "ka_i") # recuperando localizacao da variavel ka na GPU
    glUniform1f(loc_ka, 0.01) ### envia ka pra gpu
    
    loc_kd = glGetUniformLocation(program, "kd_i") # recuperando localizacao da variavel kd na GPU
    glUniform1f(loc_kd, 0.0) ### envia kd pra gpu    

    loc_ks = glGetUniformLocation(program, "ks_i") # recuperando localizacao da variavel ks na GPU
    glUniform1f(loc_ks, 0.0) ### envia ks pra gpu        
    
    loc_ns = glGetUniformLocation(program, "ns_i") # recuperando localizacao da variavel ns na GPU
    glUniform1f(loc_ns, 1.0) ### envia ns pra gpu  


def desenhaI(angle=0.0, 
            r_x=0.0, r_y=0.0, r_z=1.0,
            t_x=0.0, t_y=0.0, t_z=0.0,
            s_x=1.0, s_y=1.0, s_z=1.0,
            ka=0.8, kd=0.25, ks=0.2, ns= 32.0,
            modelDir=""):
    mat_model = model(angle, r_x, r_y, r_z, t_x, t_y, t_z, s_x, s_y, s_z)
    loc_model = glGetUniformLocation(program, "model")
    glUniformMatrix4fv(loc_model, 1, GL_TRUE, mat_model)

    #### define parametros de ilumincao do modelo
    (ka, kd, ks) = sum_k(ka, kd, ks)
    set_zero_e()
    
    loc_ka = glGetUniformLocation(program, "ka_i") # recuperando localizacao da variavel ka na GPU
    glUniform1f(loc_ka, ka) ### envia ka pra gpu
    
    loc_kd = glGetUniformLocation(program, "kd_i") # recuperando localizacao da variavel kd na GPU
    glUniform1f(loc_kd, kd) ### envia kd pra gpu    

    loc_ks = glGetUniformLocation(program, "ks_i") # recuperando localizacao da variavel ks na GPU
    glUniform1f(loc_ks, ks) ### envia ks pra gpu        
    
    loc_ns = glGetUniformLocation(program, "ns_i") # recuperando localizacao da variavel ns na GPU
    glUniform1f(loc_ns, ns) ### envia ns pra gpu        



    
    #define id da textura do modelo
    glBindTexture(GL_TEXTURE_2D, vertices_dict[modelDir][2])
    # desenha o modelo
    glDrawArrays(GL_TRIANGLES, vertices_dict[modelDir][0], vertices_dict[modelDir][1] - vertices_dict[modelDir][0]) ## renderizando

def desenhaE(angle=0.0, 
            r_x=0.0, r_y=0.0, r_z=1.0,
            t_x=0.0, t_y=0.0, t_z=0.0,
            s_x=1.0, s_y=1.0, s_z=1.0,
            ka=0.8, kd=0.25, ks=0.2, ns= 32.0,
            modelDir=""):
    mat_model = model(angle, r_x, r_y, r_z, t_x, t_y, t_z, s_x, s_y, s_z)
    loc_model = glGetUniformLocation(program, "model")
    glUniformMatrix4fv(loc_model, 1, GL_TRUE, mat_model)

    #### define parametros de ilumincao do modelo
    (ka, kd, ks) = sum_k(ka, kd, ks)
    set_zero_i()
    loc_ka = glGetUniformLocation(program, "ka_e") # recuperando localizacao da variavel ka na GPU
    glUniform1f(loc_ka, ka) ### envia ka pra gpu
    
    loc_kd = glGetUniformLocation(program, "kd_e") # recuperando localizacao da variavel kd na GPU
    glUniform1f(loc_kd, kd) ### envia kd pra gpu    

    loc_ks = glGetUniformLocation(program, "ks_e") # recuperando localizacao da variavel ks na GPU
    glUniform1f(loc_ks, ks) ### envia ks pra gpu        
    
    loc_ns = glGetUniformLocation(program, "ns_e") # recuperando localizacao da variavel ns na GPU
    glUniform1f(loc_ns, ns) ### envia ns pra gpu        



    
    #define id da textura do modelo
    glBindTexture(GL_TEXTURE_2D, vertices_dict[modelDir][2])
    # desenha o modelo
    glDrawArrays(GL_TRIANGLES, vertices_dict[modelDir][0], vertices_dict[modelDir][1] - vertices_dict[modelDir][0]) ## renderizando

def desenhaM2I(angle=0.0, 
            r_x=0.0, r_y=0.0, r_z=1.0,
            t_x=0.0, t_y=0.0, t_z=0.0,
            s_x=1.0, s_y=1.0, s_z=1.0,
            ka=0.8, kd=0.25, ks=0.7, ns= 36.0,
            modelDir=""):
    mat_model = model2(angle, r_x, r_y, r_z, t_x, t_y, t_z, s_x, s_y, s_z)
    loc_model = glGetUniformLocation(program, "model")
    glUniformMatrix4fv(loc_model, 1, GL_TRUE, mat_model)
    #define id da textura do modelo
    (ka, kd, ks) = sum_k(ka, kd, ks)
    set_zero_e()
    loc_ka = glGetUniformLocation(program, "ka_i") # recuperando localizacao da variavel ka na GPU
    glUniform1f(loc_ka, ka) ### envia ka pra gpu
    
    loc_kd = glGetUniformLocation(program, "kd_i") # recuperando localizacao da variavel kd na GPU
    glUniform1f(loc_kd, kd) ### envia kd pra gpu    

    loc_ks = glGetUniformLocation(program, "ks_i") # recuperando localizacao da variavel ks na GPU
    glUniform1f(loc_ks, ks) ### envia ks pra gpu        
    
    loc_ns = glGetUniformLocation(program, "ns_i") # recuperando localizacao da variavel ns na GPU
    glUniform1f(loc_ns, ns) ### envia ns pra gpu        
    
        
    

    glBindTexture(GL_TEXTURE_2D, vertices_dict[modelDir][2])
    # desenha o modelo
    glDrawArrays(GL_TRIANGLES, vertices_dict[modelDir][0], vertices_dict[modelDir][1] - vertices_dict[modelDir][0]) ## renderizando

def desenhaM2E(angle=0.0, 
            r_x=0.0, r_y=0.0, r_z=1.0,
            t_x=0.0, t_y=0.0, t_z=0.0,
            s_x=1.0, s_y=1.0, s_z=1.0,
            ka=0.8, kd=0.25, ks=0.7, ns= 36.0,
            modelDir=""):
    mat_model = model2(angle, r_x, r_y, r_z, t_x, t_y, t_z, s_x, s_y, s_z)
    loc_model = glGetUniformLocation(program, "model")
    glUniformMatrix4fv(loc_model, 1, GL_TRUE, mat_model)
    #define id da textura do modelo
    (ka, kd, ks) = sum_k(ka, kd, ks)
    set_zero_i()

    loc_ka = glGetUniformLocation(program, "ka_e") # recuperando localizacao da variavel ka na GPU
    glUniform1f(loc_ka, ka) ### envia ka pra gpu
    
    loc_kd = glGetUniformLocation(program, "kd_e") # recuperando localizacao da variavel kd na GPU
    glUniform1f(loc_kd, kd) ### envia kd pra gpu    

    loc_ks = glGetUniformLocation(program, "ks_e") # recuperando localizacao da variavel ks na GPU
    glUniform1f(loc_ks, ks) ### envia ks pra gpu        
    
    loc_ns = glGetUniformLocation(program, "ns_e") # recuperando localizacao da variavel ns na GPU
    glUniform1f(loc_ns, ns) ### envia ns pra gpu        
    
        
    

    glBindTexture(GL_TEXTURE_2D, vertices_dict[modelDir][2])
    # desenha o modelo
    glDrawArrays(GL_TRIANGLES, vertices_dict[modelDir][0], vertices_dict[modelDir][1] - vertices_dict[modelDir][0]) ## renderizando

def desenha_chair(angle=0.0, 
            r_x=0.0, r_y=0.0, r_z=1.0,
            t_x=0.0, t_y=0.0, t_z=0.0,
            s_x=1.0, s_y=1.0, s_z=1.0,
            ka=0.8, kd=0.25, ks=0.2, ns= 36.0,
            modelDir=""):
    mat_model = model2(angle, r_x, r_y, r_z, t_x, t_y, t_z, s_x, s_y, s_z)
    loc_model = glGetUniformLocation(program, "model")
    glUniformMatrix4fv(loc_model, 1, GL_TRUE, mat_model)
    (ka, kd, ks) = sum_k(ka, kd, ks)
    
    set_zero_e()
    
    loc_ka = glGetUniformLocation(program, "ka_i") # recuperando localizacao da variavel ka na GPU
    glUniform1f(loc_ka, ka) ### envia ka pra gpu
    
    loc_kd = glGetUniformLocation(program, "kd_i") # recuperando localizacao da variavel kd na GPU
    glUniform1f(loc_kd, kd) ### envia kd pra gpu    

    loc_ks = glGetUniformLocation(program, "ks_i") # recuperando localizacao da variavel ks na GPU
    glUniform1f(loc_ks, ks) ### envia ks pra gpu        
    
    loc_ns = glGetUniformLocation(program, "ns_i") # recuperando localizacao da variavel ns na GPU
    glUniform1f(loc_ns, ns) ### envia ns pra gpu        


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
    
        
def desenha_sofa(angle=0.0, 
            r_x=0.0, r_y=0.0, r_z=1.0,
            t_x=0.0, t_y=0.0, t_z=0.0,
            s_x=1.0, s_y=1.0, s_z=1.0,
            ka=0.8, kd=0.25, ks=0.2, ns= 36.0,
            modelDir="sofa"):
    mat_model = model(angle, r_x, r_y, r_z, t_x, t_y, t_z, s_x, s_y, s_z)
    loc_model = glGetUniformLocation(program, "model")
    glUniformMatrix4fv(loc_model, 1, GL_TRUE, mat_model)

    (ka, kd, ks) = sum_k(ka, kd, ks)
    
    set_zero_e()

    loc_ka = glGetUniformLocation(program, "ka_i") # recuperando localizacao da variavel ka na GPU
    glUniform1f(loc_ka, ka) ### envia ka pra gpu
    
    loc_kd = glGetUniformLocation(program, "kd_i") # recuperando localizacao da variavel kd na GPU
    glUniform1f(loc_kd, kd) ### envia kd pra gpu    

    loc_ks = glGetUniformLocation(program, "ks_i") # recuperando localizacao da variavel ks na GPU
    glUniform1f(loc_ks, ks) ### envia ks pra gpu        
    
    loc_ns = glGetUniformLocation(program, "ns_i") # recuperando localizacao da variavel ns na GPU
    glUniform1f(loc_ns, ns) ### envia ns pra gpu   

    
    #define id da textura do modelo
    glBindTexture(GL_TEXTURE_2D, vertices_dict[modelDir][0])
    c = 87172 * 3
    
    glDrawArrays(GL_TRIANGLES, 535947, c) ## renderizando
    c = 535947 + c
    #define id da textura do modelo
    glBindTexture(GL_TEXTURE_2D, vertices_dict[modelDir][1])
    glDrawArrays(GL_TRIANGLES, c, 837771 - c) ## renderizando






cameraPos   = glm.vec3( 0.0,      8.0,   0.0);
cameraFront = glm.vec3( 0.0,    0.0,     0.0);
cameraUp    = glm.vec3(0.0,  1.0,  0.0);


polygonal_mode = False

ka_add = 0.0
kd_add = 0.0
ks_add = 0.0

def key_event(window,key,scancode,action,mods):
    global cameraPos, cameraFront, cameraUp, polygonal_mode
    global ka_add, kd_add, ks_add

    loc_light_pos = glGetUniformLocation(program, "lightPos1") # recuperando localizacao da variavel lightPos na GPU
    print( "loc_light_pos : ", loc_light_pos);
    
    if key == 49 and (action==1 or action==2): # tecla 1
        ka_add  += 0.01
    if key == 50 and (action==1 or action==2): # tecla 2
        ka_add -= 0.01
            
    
    if key == 51 and (action==1 or action==2): # tecla 3
        kd_add += 0.01
    if key == 52 and (action==1 or action==2): # tecla 4
        kd_add -= 0.01
    

    if key == 53 and (action==1 or action==2): # tecla 5
        ks_add += 0.01
    if key == 54 and (action==1 or action==2): # tecla 6
        ks_add -= 0.01
    
    print("ka : " , ka_add)
    print("kd : " , kd_add)
    print("ks : " , ks_add)



    cameraSpeed = 1.0
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
monstro_dance = 0.02



glfw.show_window(window)
glfw.set_cursor_pos(window, lastX, lastY)
glEnable(GL_DEPTH_TEST) ### importante para 3D
   

    
while not glfw.window_should_close(window):

    glfw.poll_events() 
    
    
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    
    glClearColor(0.2, 0.2, 0.2, 1.0)
    
    if polygonal_mode==True:
        glPolygonMode(GL_FRONT_AND_BACK,GL_LINE)
    if polygonal_mode==False:
        glPolygonMode(GL_FRONT_AND_BACK,GL_FILL)
    
     # interior da casa
    
#####################################################################################################


    # ka=0.8, kd=0.25, ks=0.2, ns= 36.0,

    # interior da casa


    # desenhaI(s_x=0.005, s_y=0.005, s_z=0.005,t_x =-4, t_z=-19.8,ka=1.0,  kd=0.1,ks=0.0, modelDir="aya");
    # desenhaI(s_x=20, s_y=20, s_z=20, t_z = -10,t_y= 0.0001, ka= 1.0,kd=0.2, ks=0.6 ,modelDir="floor")
    # desenhaI(s_x=6, s_y=6, s_z=6, t_x = -14,t_z= 13, ka= 1.0,kd=0.0, ks=0.0,modelDir="floor")    
    # desenha_chair(s_x=0.04, s_y=0.04, s_z=0.04,t_x =-4 ,t_z=-20, kd=0.5, ks=0.0 ,modelDir="chair");
    # desenhaM2I(angle=-90,r_x=1.0 ,r_z=0.0,s_x=0.05, s_y=0.05, s_z=0.05,t_x=0.0,kd=0.7,modelDir="beagle")
    # desenhaI(s_x=3.58, s_y=5, s_z=4,t_x= -0.7, t_y = -2,t_z= -8, ks=0.0, modelDir="cottage")
    # desenhaM2I(angle=-90, r_y=1.0 ,r_z=0.0, s_x=0.013, s_y=0.013, s_z=0.013, t_y= 3, t_x=18, t_z=-18,ka=1.0,kd=0.5, ks=0.3 ,modelDir="table1")
    # desenha_sofa(s_x=0.05, s_y=0.05, s_z=0.05, t_x=-17, t_y = 4,t_z=-22, ka=0.8, kd=0.3, ks=0.0 ,modelDir="sofa")
    # desenhaI(s_x=2, s_y=0.5, s_z=2, t_x=-13.5, t_z = -14, ka=0.95 ,kd=0.0, ks=1.0 , ns=36.0, modelDir="stool")
    # desenhaI(s_x=0.32, s_y=0.3, s_z=0.32,t_x=-13.5, t_y= 2.2, t_z= -14,modelDir="caixa")
    # desenha_lamp(angle=180, r_x=1.0,r_z=0.0, s_x=0.3, s_y=0.3, s_z=0.3,t_x=-13.5, t_y= 2.7, t_z= -14,modelDir="luz")
    # desenhaI(s_x=0.8, s_y=0.8, s_z=0.8,t_y= 0, t_x=18, t_z=-13,  ka=1.0,kd=0.1,ks=0.0,modelDir="plant")
    # desenha_chair(angle=90, r_y=1.0 ,r_z=0.0,s_x=0.04, s_y=0.04, s_z=0.04,  t_x=16, t_z=-16, kd=0.5,ks=0.0,modelDir="chair");
    # desenhaM2I(angle=180, r_y=1.0,r_z=0.0,s_x=0.08, s_y=0.03, s_z=0.05,t_x=-13.5,t_y=1.0 ,t_z = -2.3, kd=0.4,ks=0.5, ns=30,  modelDir="cabinet")
    # desenhaM2I(angle=180,r_y=1.0,r_z=0.0,s_x=0.5,s_y=0.5,s_z=0.5,t_x= 18, t_z=3, t_y= 0.0, ka=1.0, kd=0.2,ks=0.0,modelDir="bed1")
    # desenhaM2I(angle=90,r_y=0.1,r_z=0.0,  s_x=0.04, s_y=0.06, s_z=0.08,t_y=2.0,t_x=-20.0, t_z = 12.5,modelDir="cabinet")
    # desenhaI(s_x=0.4, s_y=0.4, s_z=0.4, t_y=4.3,t_x=-18, t_z= 7,  ka=0.8,kd=0.6, ks=0.0, modelDir="plant")
    # desenhaI(s_x=0.4, s_y=0.4, s_z=0.4, t_y=4.3,t_x=-18, t_z= 16, ka=0.8,kd=0.6, ks=0.0, modelDir="plant")
    # desenhaM2I(angle=-90,r_y=0.1,r_z=0.0,  s_x=1.1, s_y=1.1, s_z=1.1,t_x=-16,  t_z = 12.5,ka=1.0, kd=0.4,modelDir="chair2")
    # desenhaM2I(angle=-90,r_y=0.1,r_z=0.0,  s_x=1.1, s_y=1.1, s_z=1.1,t_x=-16,  t_z = 10,ka=1.0, kd=0.4 ,modelDir="chair2")
    # desenhaM2I(angle=180, r_y=1.0,r_z=0.0, t_x=-13.5, t_y= 2.1,t_z=-4, ka=0.5, kd=0.8,ks=1.0,ns=56.0, modelDir="tv")

#####################################################################################################


    #exterior da casa


    # desenhaE(s_x=140, s_y=140, s_z=140, t_z=-40 , t_y = -2,ka=0.6, kd=0.0,modelDir="grass")
    # desenhaE(s_x=10, s_z= 20, t_z= -50, t_y= -1.8,modelDir= "cobleStone")
    # desenhaE(s_x=140, s_y=140, s_z=140, t_z=-40 , t_y = 90, ka=0.1, kd=1.0, ks=1.0, ns=40.0 , modelDir="sky")  

    # if  sun_angle > 360:
    #     sun_angle = 0.0
    # sun_angle += 0.3;

    # desenha_sun(angle =sun_angle * 3, r_y=1.0, r_z=0.0, s_x= 1.0 ,s_y=1.0 ,s_z=1.0, t_z = -70 + 100 * math.sin(math.radians(sun_angle)) + 50, t_x = 100 * math.cos(math.radians(sun_angle)) ,t_y = 80, modelDir="sun")

    # desenhaE(angle = 90 , r_z= 0 ,r_y = 1,s_x=140.0, s_z=14.0, t_z = -40, t_x = 40 ,t_y = -1.8,ka=0.5, ks=1.0, ns=60, modelDir="street")
    # desenhaE(angle = 90 , r_z= 0 ,r_y = 1,s_x=140.0, s_z=20.0, t_z = 50, t_x = 40 ,t_y = -1.8, ka=1.0, kd=0.0,ks=0.0, modelDir="water")
  
    # # desenhando golfinhos

    # if  dolphin_position < -150:
    #     dolphin_position = 40
    # dolphin_position -= 0.19
    # ydp = fish_move(dolphin_position, 5, 40, True)
    # desenhaM2E(angle= angle_dolphin,r_x=1.0,r_z = 0.0 ,s_x=1.1, s_y=1.1, s_z=1.1,t_z = dolphin_position , t_x = 45 ,t_y = -2.0 + ydp , ka=0.8, kd=0.2,  ks=1.0, ns=70.0, modelDir="dolphin")
    # desenhaM2E(angle= angle_dolphin,r_x=1.0,r_z = 0.0 ,s_x=1.1, s_y=1.1, s_z=1.1,t_z = dolphin_position + 10, t_x = 40 ,t_y = -2.0 + ydp , ka=0.8, kd=0.2,  ks=1.0, ns=70.0, modelDir="dolphin")

    # # desenhando baleia

    # if  whale_position < -150:
    #     whale_position = 40
    # whale_position -= 0.1
    # ydw = fish_move(whale_position, 1, 40, False)
    # desenhaM2E(angle=90,r_x=0.0,r_y = 1, r_z = 0 ,s_x=0.2, s_y=0.2, s_z=0.2,t_z = whale_position , t_x = 55 ,t_y = -2.5 + ydw , ka=0.6, kd=0.4, ks=1.0, ns=70.0 , modelDir="whale")


    # # desenhando os aviões
    # anglePlane += 1
    # if anglePlane > 360:
    #     anglePlane = 0.0

    # anglePlane2 = anglePlane + 90
    # (planex1, planey2) = elipse( 50 , 40, anglePlane)
    # desenhaM2E(angle=anglePlane,r_y=1.0,r_z=0,t_y= 40 + 10 * math.sin(math.radians(1.5 * anglePlane)), t_x = 5  + planex1, t_z=-70 + planey2,ka=0.7, kd=0.5, ks=0.0 ,  modelDir="plane1")
    
    # (planex1, planey2) = elipse( 50 , 40, anglePlane2)
    # desenhaM2E(angle=anglePlane2, s_z=0.7, s_y=0.7, s_x=0.7 ,r_y=1.0,r_z=0,t_y= 40 + 8 * math.sin(math.radians(1.5 * anglePlane)), t_x = 5  + planex1, t_z=-70 + planey2, ka=0.7, kd=0.5, ks=0.9, ns=60.0 ,modelDir="plane2")
    

    
    # # desenha container
    # desenhaM2E( s_z=0.8,s_y=0.6,s_x=0.8, t_y=4.8,t_x=5 , t_z= -100 ,ka=0.7, kd=0.5, ks=1.0, ns=150.0, modelDir="container")
    if monstro_dance > 0:
        monstro_dance = -0.02
    else :
        monstro_dance = 0.02

    desenhaM2E( s_z=0.5 + monstro_dance,s_y=0.5 + monstro_dance,s_x=0.5 + monstro_dance, t_y=-1.0,t_x=-2 , t_z= -90 ,ka=1.0,kd=0.3, modelDir="librarian")
    desenhaM2E( s_z=2,s_y=1.2,s_x=3, t_y=0.0,t_x=5 , t_z= -90 ,ka=1.0,kd=0.3, modelDir="caixa2")
    desenhaM2E(angle=-90,r_y=1.0,r_z=0.0, s_z=1.3 + monstro_dance,s_y=1.3 + monstro_dance,s_x=1.3 + monstro_dance, t_y=1.5,t_x=4.5, t_z= -91 ,ka=1.0,kd=0.3, modelDir="monstro")
    desenhaM2E(angle=180,r_y=1.0,r_z=0.0, s_z=0.5 + monstro_dance,s_y=0.5 + monstro_dance,s_x=0.5 + monstro_dance, t_y=-1.0,t_x=12, t_z= -90 , ka=1.0, kd=0.3, modelDir="librarian")


    # # Faz a bola quicar
    # if ball_position > 3:
    #     x_ball_postion = -0.1;
    # elif ball_position < -3:
    #     x_ball_postion = 0.1;
    # ball_position += x_ball_postion

    # desenhaE(s_x=0.2, s_y=0.2, s_z=0.2,t_z= -55, t_x = 20 , t_y= ball_move(),  ka=1.0, kd=0.3, ks=0.9, ns=40.0, modelDir="ball")

    
    # # movendo o carro
    # if car_position < -125:
    #     car_position = 40
    # car_position -= 1

    # desenhaE(s_x=3.0, s_y=3.0 ,s_z=3.0, t_z = car_position, t_x = -33 ,t_y = -1.8,  ka=1.0, kd=0.3, ks=0.9, ns=40.0,  modelDir="car")    









#####################################################################################################

    mat_view = view()
    loc_view = glGetUniformLocation(program, "view")
    glUniformMatrix4fv(loc_view, 1, GL_FALSE, mat_view)

    mat_projection = projection()
    loc_projection = glGetUniformLocation(program, "projection")
    glUniformMatrix4fv(loc_projection, 1, GL_FALSE, mat_projection)    
    
    # atualizando a posicao da camera/observador na GPU para calculo da reflexao especular
    loc_view_pos = glGetUniformLocation(program, "viewPos") # recuperando localizacao da variavel viewPos na GPU
    glUniform3f(loc_view_pos, cameraPos[0], cameraPos[1], cameraPos[2]) ### posicao da camera/observador (x,y,z)
    
    glfw.swap_buffers(window)

glfw.terminate()