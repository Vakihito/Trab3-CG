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

    // calculando reflexao ambiente interior
    vec3 ambient = ka_i * lightColor;             

    ////////////////////////
    // Luz #1
    ////////////////////////
    
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