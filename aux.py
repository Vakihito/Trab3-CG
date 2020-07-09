             ////////////////////////
            // Luz #3
            ////////////////////////

            // calculando reflexao ambiente interior
            vec3 ambient_ic = ka_ic * lightColor2; 
            
            // calculando reflexao difusa
            vec3 norm3 = normalize(out_normal); // normaliza vetores perpendiculares
            vec3 lightDir3 = normalize(lightPos2 - out_fragPos); // direcao da luz
            float diff3 = max(dot(norm3, lightDir3), 0.0); // verifica limite angular (entre 0 e 90)
            vec3 diffuse3 = kd_ic * diff3 * lightColor2; // iluminacao difusa
            
            // calculando reflexao especular
            vec3 viewDir3 = normalize(viewPos - out_fragPos); // direcao do observador/camera
            vec3 reflectDir3 = reflect(-lightDir3, norm3); // direcao da reflexao
            float spec3 = pow(max(dot(viewDir3, reflectDir3), 0.0), ns_ic);
            vec3 specular3 = ks_ic * spec3 * lightColor2;    