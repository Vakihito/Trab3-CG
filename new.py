import glfw
from OpenGL.GL import *
import OpenGL.GL.shaders
import numpy as np
import glm
import math
from PIL import Image

glfw.init()
glfw.window_hint(glfw.VISIBLE, glfw.FALSE);
altura = 1600
largura = 1200
window = glfw.create_window(largura, altura, "Iluminação", None, None)
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

        // parametros da iluminacao ambiente e difusa
        uniform vec3 lightPos1; // define coordenadas de posicao da luz #1
        uniform vec3 lightPos2; // define coordenadas de posicao da luz #2
        uniform float ka; // coeficiente de reflexao ambiente
        uniform float kd; // coeficiente de reflexao difusa
        
        // parametros da iluminacao especular
        uniform vec3 viewPos; // define coordenadas com a posicao da camera/observador
        uniform float ks; // coeficiente de reflexao especular
        uniform float ns; // expoente de reflexao especular
        
        // parametro com a cor da(s) fonte(s) de iluminacao
        vec3 lightColor = vec3(1.0, 1.0, 1.0);

        // parametros recebidos do vertex shader
        varying vec2 out_texture; // recebido do vertex shader
        varying vec3 out_normal; // recebido do vertex shader
        varying vec3 out_fragPos; // recebido do vertex shader
        uniform sampler2D samplerTexture;
        
        
        
        void main(){
        
            // calculando reflexao ambiente
            vec3 ambient = ka * lightColor;             
        
            ////////////////////////
            // Luz #1
            ////////////////////////
            
            // calculando reflexao difusa
            vec3 norm1 = normalize(out_normal); // normaliza vetores perpendiculares
            vec3 lightDir1 = normalize(lightPos1 - out_fragPos); // direcao da luz
            float diff1 = max(dot(norm1, lightDir1), 0.0); // verifica limite angular (entre 0 e 90)
            vec3 diffuse1 = kd * diff1 * lightColor; // iluminacao difusa
            
            // calculando reflexao especular
            vec3 viewDir1 = normalize(viewPos - out_fragPos); // direcao do observador/camera
            vec3 reflectDir1 = reflect(-lightDir1, norm1); // direcao da reflexao
            float spec1 = pow(max(dot(viewDir1, reflectDir1), 0.0), ns);
            vec3 specular1 = ks * spec1 * lightColor;    
            
            
            ////////////////////////
            // Luz #2
            ////////////////////////
            
            // calculando reflexao difusa
            vec3 norm2 = normalize(out_normal); // normaliza vetores perpendiculares
            vec3 lightDir2 = normalize(lightPos2 - out_fragPos); // direcao da luz
            float diff2 = max(dot(norm2, lightDir2), 0.0); // verifica limite angular (entre 0 e 90)
            vec3 diffuse2 = kd * diff2 * lightColor; // iluminacao difusa
            
            // calculando reflexao especular
            vec3 viewDir2 = normalize(viewPos - out_fragPos); // direcao do observador/camera
            vec3 reflectDir2 = reflect(-lightDir2, norm2); // direcao da reflexao
            float spec2 = pow(max(dot(viewDir2, reflectDir2), 0.0), ns);
            vec3 specular2 = ks * spec2 * lightColor;    
            
            ////////////////////////
            // Combinando as duas fontes
            ////////////////////////
            
            // aplicando o modelo de iluminacao
            vec4 texture = texture2D(samplerTexture, out_texture);
            vec4 result = vec4((ambient + diffuse1 + diffuse2 + specular1 + specular2),1.0) * texture; // aplica iluminacao
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
qtd_texturas = 90
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




# exterior da casa
################################################################

processObjects("grass", "terreno.obj", "grass.jpeg")
# processObjects_no_normal("street", "terreno2.obj", "street.jpg")
# processObjects_no_normal("water", "water.obj", "water5.jpg")
# processObjects("car", "car.obj", "car.jpg")
# processObjects("dogh", "doghouse.obj", "2_BaseColor.jpg")
# processObjects("ball", "ball.obj", "ball.jpg")
# processObjects("doberman", "dog2.obj", "Doberman_Pinscher_dif.jpg")
# processObjects("cat", "cat.obj", "Cat_bump.jpg")
# processObjects("sky", "terreno.obj", "sky2.jpg")
# processObjects_no_normal("cobleStone", "floor.obj", "cobleStone.jpg")
# processObjects("dolphin", "dolphin1.obj", "dolphin.jpg")
processObjects("whale", "whale2.obj", "10054_Whale_Diffuse_v2.jpg")
# processObjects("plane1", "plane1.obj", "plane1.jpg")
# processObjects("plane2", "plane2.obj", "plane2.jpg")
# processObjects("container", "container.obj", "12281_Container_diffuse.jpg")
# processObjects("librarian", "librarian1.obj", "act_bibliotekar.jpg")
processObjects("sun", "sun2.obj", "sun.jpg")



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

def desenha_caixa():
    

    # aplica a matriz model
    angle = 0.0
    
    r_x = 0.0; r_y = 1.0; r_z = 0.0;
    
    # translacao
    t_x = 0.0; t_y = 0.0; t_z = 0.0;
    
    # escala
    s_x = 1.0; s_y = 1.0; s_z = 1.0;
    
    mat_model = model(angle, r_x, r_y, r_z, t_x, t_y, t_z, s_x, s_y, s_z)
    loc_model = glGetUniformLocation(program, "model")
    glUniformMatrix4fv(loc_model, 1, GL_TRUE, mat_model)
       
    
    #### define parametros de ilumincao do modelo
    ka = 0.3 # coeficiente de reflexao ambiente do modelo
    kd = 0.3 # coeficiente de reflexao difusa do modelo
    ks = 0.9 # coeficiente de reflexao especular do modelo
    ns = 64.0 # expoente de reflexao especular
    
    loc_ka = glGetUniformLocation(program, "ka") # recuperando localizacao da variavel ka na GPU
    glUniform1f(loc_ka, ka) ### envia ka pra gpu
    
    loc_kd = glGetUniformLocation(program, "kd") # recuperando localizacao da variavel kd na GPU
    glUniform1f(loc_kd, kd) ### envia kd pra gpu    
    
    loc_ks = glGetUniformLocation(program, "ks") # recuperando localizacao da variavel ks na GPU
    glUniform1f(loc_ks, ks) ### envia ks pra gpu        
    
    loc_ns = glGetUniformLocation(program, "ns") # recuperando localizacao da variavel ns na GPU
    glUniform1f(loc_ns, ns) ### envia ns pra gpu        

    
    #define id da textura do modelo
    glBindTexture(GL_TEXTURE_2D, 0)
    
    
    # desenha o modelo
    glDrawArrays(GL_TRIANGLES, 0, 36) ## renderizando
def desenha_luz1(t_x, t_y, t_z):
    

    # aplica a matriz model
    angle = 0.0
    
    r_x = 0.0; r_y = 0.0; r_z = 1.0;
    
    # translacao
    #t_x = 0.0; t_y = 0.0; t_z = 0.0;
    
    # escala
    s_x = 0.1; s_y = 0.1; s_z = 0.1;
    
    mat_model = model(angle, r_x, r_y, r_z, t_x, t_y, t_z, s_x, s_y, s_z)
    loc_model = glGetUniformLocation(program, "model")
    glUniformMatrix4fv(loc_model, 1, GL_TRUE, mat_model)
       
    
    #### define parametros de ilumincao do modelo
    ka = 1 # coeficiente de reflexao ambiente do modelo
    kd = 1 # coeficiente de reflexao difusa do modelo
    ks = 1 # coeficiente de reflexao especular do modelo
    ns = 1000.0 # expoente de reflexao especular
    
    loc_ka = glGetUniformLocation(program, "ka") # recuperando localizacao da variavel ka na GPU
    glUniform1f(loc_ka, ka) ### envia ka pra gpu
    
    loc_kd = glGetUniformLocation(program, "kd") # recuperando localizacao da variavel kd na GPU
    glUniform1f(loc_kd, kd) ### envia kd pra gpu    
    
    loc_ks = glGetUniformLocation(program, "ks") # recuperando localizacao da variavel ks na GPU
    glUniform1f(loc_ks, ks) ### envia ns pra gpu        
    
    loc_ns = glGetUniformLocation(program, "ns") # recuperando localizacao da variavel ns na GPU
    glUniform1f(loc_ns, ns) ### envia ns pra gpu            
    
    loc_light_pos = glGetUniformLocation(program, "lightPos1") # recuperando localizacao da variavel lightPos na GPU
    glUniform3f(loc_light_pos, t_x, t_y, t_z) ### posicao da fonte de luz
        
    
    #define id da textura do modelo
    glBindTexture(GL_TEXTURE_2D, 1)
    
    
    # desenha o modelo
    glDrawArrays(GL_TRIANGLES, 36, 36) ## renderizando
    

def desenha_luz2(t_x, t_y, t_z):
    

    # aplica a matriz model
    angle = 0.0
    
    r_x = 0.0; r_y = 0.0; r_z = 1.0;
    
    # translacao
    #t_x = 0.0; t_y = 0.0; t_z = 0.0;
    
    # escala
    s_x = 0.1; s_y = 0.1; s_z = 0.1;
    
    mat_model = model(angle, r_x, r_y, r_z, t_x, t_y, t_z, s_x, s_y, s_z)
    loc_model = glGetUniformLocation(program, "model")
    glUniformMatrix4fv(loc_model, 1, GL_TRUE, mat_model)
       
    
    #### define parametros de ilumincao do modelo
    ka = 1 # coeficiente de reflexao ambiente do modelo
    kd = 1 # coeficiente de reflexao difusa do modelo
    ks = 1 # coeficiente de reflexao especular do modelo
    ns = 1000.0 # expoente de reflexao especular
    
    loc_ka = glGetUniformLocation(program, "ka") # recuperando localizacao da variavel ka na GPU
    glUniform1f(loc_ka, ka) ### envia ka pra gpu
    
    loc_kd = glGetUniformLocation(program, "kd") # recuperando localizacao da variavel kd na GPU
    glUniform1f(loc_kd, kd) ### envia kd pra gpu    
    
    loc_ks = glGetUniformLocation(program, "ks") # recuperando localizacao da variavel ks na GPU
    glUniform1f(loc_ks, ks) ### envia ns pra gpu        
    
    loc_ns = glGetUniformLocation(program, "ns") # recuperando localizacao da variavel ns na GPU
    glUniform1f(loc_ns, ns) ### envia ns pra gpu            
    
    loc_light_pos = glGetUniformLocation(program, "lightPos2") # recuperando localizacao da variavel lightPos na GPU
    glUniform3f(loc_light_pos, t_x, t_y, t_z) ### posicao da fonte de luz
        
    
    #define id da textura do modelo
    glBindTexture(GL_TEXTURE_2D, 2)
    
    
    # desenha o modelo
    glDrawArrays(GL_TRIANGLES, 72, 36) ## renderizando

ka_chaleira = 0.8
kd_chaleira = 0.25
ks_chaleira = 0.4
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
    
    loc_ka = glGetUniformLocation(program, "ka") # recuperando localizacao da variavel ka na GPU
    glUniform1f(loc_ka, ka) ### envia ka pra gpu
    
    loc_kd = glGetUniformLocation(program, "kd") # recuperando localizacao da variavel kd na GPU
    glUniform1f(loc_kd, kd) ### envia kd pra gpu    
    
    loc_ks = glGetUniformLocation(program, "ks") # recuperando localizacao da variavel ks na GPU
    glUniform1f(loc_ks, ks) ### envia ns pra gpu        
    
    loc_ns = glGetUniformLocation(program, "ns") # recuperando localizacao da variavel ns na GPU
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
    
    loc_ka = glGetUniformLocation(program, "ka") # recuperando localizacao da variavel ka na GPU
    glUniform1f(loc_ka, ka) ### envia ka pra gpu
    
    loc_kd = glGetUniformLocation(program, "kd") # recuperando localizacao da variavel kd na GPU
    glUniform1f(loc_kd, kd) ### envia kd pra gpu    
    
    loc_ks = glGetUniformLocation(program, "ks") # recuperando localizacao da variavel ks na GPU
    glUniform1f(loc_ks, ks) ### envia ns pra gpu        
    
    loc_ns = glGetUniformLocation(program, "ns") # recuperando localizacao da variavel ns na GPU
    glUniform1f(loc_ns, ns) ### envia ns pra gpu            
    
    loc_light_pos = glGetUniformLocation(program, "lightPos1") # recuperando localizacao da variavel lightPos na GPU
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


def desenha(angle=0.0, 
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
    
    loc_ka = glGetUniformLocation(program, "ka") # recuperando localizacao da variavel ka na GPU
    glUniform1f(loc_ka, ka) ### envia ka pra gpu
    
    loc_kd = glGetUniformLocation(program, "kd") # recuperando localizacao da variavel kd na GPU
    glUniform1f(loc_kd, kd) ### envia kd pra gpu    

    loc_ks = glGetUniformLocation(program, "ks") # recuperando localizacao da variavel ks na GPU
    glUniform1f(loc_ks, ks) ### envia ks pra gpu        
    
    loc_ns = glGetUniformLocation(program, "ns") # recuperando localizacao da variavel ns na GPU
    glUniform1f(loc_ns, ns) ### envia ns pra gpu        



    
    #define id da textura do modelo
    glBindTexture(GL_TEXTURE_2D, vertices_dict[modelDir][2])
    # desenha o modelo
    glDrawArrays(GL_TRIANGLES, vertices_dict[modelDir][0], vertices_dict[modelDir][1] - vertices_dict[modelDir][0]) ## renderizando

def desenha_no_light(angle=0.0, 
            r_x=0.0, r_y=0.0, r_z=1.0,
            t_x=0.0, t_y=0.0, t_z=0.0,
            s_x=1.0, s_y=1.0, s_z=1.0,
            ka=0.8, kd=0.25, ks=0.2, ns= 36.0,
            modelDir=""):
    mat_model = model(angle, r_x, r_y, r_z, t_x, t_y, t_z, s_x, s_y, s_z)
    loc_model = glGetUniformLocation(program, "model")
    glUniformMatrix4fv(loc_model, 1, GL_TRUE, mat_model)
    #define id da textura do modelo
    (ka, kd, ks) = sum_k(ka, kd, ks)

    loc_ka = glGetUniformLocation(program, "ka") # recuperando localizacao da variavel ka na GPU
    glUniform1f(loc_ka, ka) ### envia ka pra gpu
    
    loc_kd = glGetUniformLocation(program, "kd") # recuperando localizacao da variavel kd na GPU
    glUniform1f(loc_kd, kd) ### envia kd pra gpu    

    loc_ks = glGetUniformLocation(program, "ks") # recuperando localizacao da variavel ks na GPU
    glUniform1f(loc_ks, ks) ### envia ks pra gpu        
    
    loc_ns = glGetUniformLocation(program, "ns") # recuperando localizacao da variavel ns na GPU
    glUniform1f(loc_ns, ns) ### envia ns pra gpu        


    glBindTexture(GL_TEXTURE_2D, vertices_dict[modelDir][2])
    # desenha o modelo
    glDrawArrays(GL_TRIANGLES, vertices_dict[modelDir][0], vertices_dict[modelDir][1] - vertices_dict[modelDir][0]) ## renderizando



def desenhaM2(angle=0.0, 
            r_x=0.0, r_y=0.0, r_z=1.0,
            t_x=0.0, t_y=0.0, t_z=0.0,
            s_x=1.0, s_y=1.0, s_z=1.0,
            ka=0.8, kd=0.25, ks=0.7, ns= 36.0,
            modelDir=""):
    mat_model = model2(angle, r_x, r_y, r_z, t_x, t_y, t_z, s_x, s_y, s_z)
    loc_model = glGetUniformLocation(program, "model")
    glUniformMatrix4fv(loc_model, 1, GL_TRUE, mat_model)
    #define id da textura do modelo
    (ka_aux, kd_aux, ks_aux) = sum_k(ka, kd, ks)

    loc_ka = glGetUniformLocation(program, "ka") # recuperando localizacao da variavel ka na GPU
    glUniform1f(loc_ka, ka_aux) ### envia ka pra gpu
    
    loc_kd = glGetUniformLocation(program, "kd") # recuperando localizacao da variavel kd na GPU
    glUniform1f(loc_kd, kd_aux) ### envia kd pra gpu    

    loc_ks = glGetUniformLocation(program, "ks") # recuperando localizacao da variavel ks na GPU
    glUniform1f(loc_ks, ks_aux) ### envia ks pra gpu        
    
    loc_ns = glGetUniformLocation(program, "ns") # recuperando localizacao da variavel ns na GPU
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
    mat_model = model(angle, r_x, r_y, r_z, t_x, t_y, t_z, s_x, s_y, s_z)
    loc_model = glGetUniformLocation(program, "model")
    glUniformMatrix4fv(loc_model, 1, GL_TRUE, mat_model)
    (ka, kd, ks) = sum_k(ka, kd, ks)
    
    
    loc_ka = glGetUniformLocation(program, "ka") # recuperando localizacao da variavel ka na GPU
    glUniform1f(loc_ka, ka) ### envia ka pra gpu
    
    loc_kd = glGetUniformLocation(program, "kd") # recuperando localizacao da variavel kd na GPU
    glUniform1f(loc_kd, kd) ### envia kd pra gpu    

    loc_ks = glGetUniformLocation(program, "ks") # recuperando localizacao da variavel ks na GPU
    glUniform1f(loc_ks, ks) ### envia ks pra gpu        
    
    loc_ns = glGetUniformLocation(program, "ns") # recuperando localizacao da variavel ns na GPU
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
    

    loc_ka = glGetUniformLocation(program, "ka") # recuperando localizacao da variavel ka na GPU
    glUniform1f(loc_ka, ka) ### envia ka pra gpu
    
    loc_kd = glGetUniformLocation(program, "kd") # recuperando localizacao da variavel kd na GPU
    glUniform1f(loc_kd, kd) ### envia kd pra gpu    

    loc_ks = glGetUniformLocation(program, "ks") # recuperando localizacao da variavel ks na GPU
    glUniform1f(loc_ks, ks) ### envia ks pra gpu        
    
    loc_ns = glGetUniformLocation(program, "ns") # recuperando localizacao da variavel ns na GPU
    glUniform1f(loc_ns, ns) ### envia ns pra gpu   

    
    #define id da textura do modelo
    glBindTexture(GL_TEXTURE_2D, vertices_dict[modelDir][0])
    c = 87172 * 3
    
    glDrawArrays(GL_TRIANGLES, 535947, c) ## renderizando
    c = 535947 + c
    #define id da textura do modelo
    glBindTexture(GL_TEXTURE_2D, vertices_dict[modelDir][1])
    glDrawArrays(GL_TRIANGLES, c, 837771 - c) ## renderizando



def desenha_chaleira():
    

    # aplica a matriz model
    angle = 0.0
    
    r_x = 0.0; r_y = 1.0; r_z = 0.0;
    
    # translacao
    t_x = 0.0; t_y = 0.0; t_z = 0.0;
    
    # escala
    s_x = 0.1; s_y = 0.1; s_z = 0.1;
    
    mat_model = model(angle, r_x, r_y, r_z, t_x, t_y, t_z, s_x, s_y, s_z)
    loc_model = glGetUniformLocation(program, "model")
    glUniformMatrix4fv(loc_model, 1, GL_TRUE, mat_model)
       
    
    #### define parametros de ilumincao do modelo
    ns = 32.0 # expoente de reflexao especular
    
    loc_ka = glGetUniformLocation(program, "ka") # recuperando localizacao da variavel ka na GPU
    glUniform1f(loc_ka, ka_chaleira) ### envia ka pra gpu
    
    loc_kd = glGetUniformLocation(program, "kd") # recuperando localizacao da variavel kd na GPU
    glUniform1f(loc_kd, kd_chaleira) ### envia kd pra gpu    
    
    loc_ks = glGetUniformLocation(program, "ks") # recuperando localizacao da variavel ks na GPU
    glUniform1f(loc_ks, ks_chaleira) ### envia ks pra gpu        
    
    loc_ns = glGetUniformLocation(program, "ns") # recuperando localizacao da variavel ns na GPU
    glUniform1f(loc_ns, ns) ### envia ns pra gpu        

    
    #define id da textura do modelo
    glBindTexture(GL_TEXTURE_2D, 4)
    
    
    # desenha o modelo
    glDrawArrays(GL_TRIANGLES, 831228, 875244-831228) ## renderizando


cameraPos   = glm.vec3( 0.0,      0.0,   0.0);
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

glfw.show_window(window)
glfw.set_cursor_pos(window, lastX, lastY)
glEnable(GL_DEPTH_TEST) ### importante para 3D
   
ang = 0.1
    
while not glfw.window_should_close(window):

    glfw.poll_events() 
    
    
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    
    glClearColor(0.2, 0.2, 0.2, 1.0)
    
    if polygonal_mode==True:
        glPolygonMode(GL_FRONT_AND_BACK,GL_LINE)
    if polygonal_mode==False:
        glPolygonMode(GL_FRONT_AND_BACK,GL_FILL)
    
    #desenha_caixa()
    desenha(s_x=140, s_y=0.001, s_z=140, t_z=-40 , t_y = -2,modelDir="grass")


    
    # desenha baleia
    sun_angle+=1;
    desenha_sun(angle =sun_angle * 3, r_y=1.0, r_z=0.0, s_x= 3.0 ,s_y=3.0 ,s_z=3.0, t_z = -70 + 100 * math.sin(math.radians(sun_angle)) + 50, t_x = 100 * math.cos(math.radians(sun_angle)) ,t_y = 80, modelDir="sun")
    
    ydw = fish_move(whale_position, 1, 40, False)
    desenhaM2(angle=90,r_y = 1.0, r_z = 0 ,s_x=0.2, s_y=0.2, s_z=0.2,t_z = -5  , t_x = 0,t_y = 0.0, ka=0.0, kd=1.0, ks=1.0,ns=50.0 , modelDir="whale")


    

    
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