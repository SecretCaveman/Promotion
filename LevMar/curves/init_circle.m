function coeff = init_circle(n, radius, center)
%Creates a circle with center "center" and radius "radius", discretized by
%n points.
    theta = (0:1:n-1) /(n)*(2*pi);
    L = 2*n*sin(pi/n)*radius;
    b = [center(1),center(2)-radius];
    coeff = [theta' ; L ; b'];
end
