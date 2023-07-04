using Dates
using Images

Base.@kwdef mutable struct Html
    num_imgs::Int = 0
    styles::String = ""
    scripts::String = ""
    body::String = ""
    head::String = "Output"
    timestamp::DateTime
    dir::String
end

function html_new(styles=atari_styles(), scripts=atari_scripts())
    timestamp = Dates.now()
    dir = "$(Dates.format(timestamp, "yyyy-mm-dd__HH-MM-SS"))"
    mkpath("out/html/$dir/imgs")
    Html(timestamp=timestamp, dir=dir, styles=styles, scripts=scripts)
end

function atari_styles()
    """
    img {
        image-rendering: pixelated
    }
    table, th, td {
        border: 1px solid black;
        border-collapse: collapse;
    }
    """
end

function atari_scripts()
    raw"""
    """
end

global curr_html::Html = html_new()

function html_get()
    curr_html
end

"""
Sets the global html, returning the old one
"""
function html_set(html::Html)
    global curr_html
    old = curr_html
    curr_html = html
    old
end

"""
Makes a fresh global html, returning the old one
"""
function html_fresh()
    html_set(html_new())
end

function html_body(body...)
    for b in body
        curr_html.body *= "\n\n" * string(b)
    end
end

function add_style!(style)
    curr_html.styles *= "\n\n" * string(style)
end



html_img(img::Array{Float64,3}, attrs...; kwargs...) = html_img(Array(colorview(RGB,img)), attrs...; kwargs...)

function html_img(img::Matrix{RGB{Float64}}, attrs...; width="200px")
    curr_html.num_imgs += 1
    path = "imgs/img_$(curr_html.num_imgs).png"
    save("out/html/$(curr_html.dir)/$path", img)
    attrs = join(string.(attrs), " ")
    if !isnothing(width)
        attrs *= " width=$width"
    end
    res = if !isempty(attrs) "<img src=$(path) $attrs>" else "<img src=$(path)>" end
    res
end

html_gif(gif::Array{Float64,4}, attrs...; kwargs...) = html_gif(Array(colorview(RGB,gif)), attrs...; kwargs...)

function html_gif(gif::Array{RGB{Float64},3}, attrs...; fps=3, width="200px")
    curr_html.num_imgs += 1
    path = "imgs/img_$(curr_html.num_imgs).gif"
    save("out/html/$(curr_html.dir)/$path", gif, fps=fps)
    attrs = join(string.(attrs), " ")
    if !isnothing(width)
        attrs *= " width=$width"
    end
    res = if !isempty(attrs) "<img src=$(path) $attrs>" else "<img src=$(path)>" end
    res
end

function html_table(table::Matrix, attrs...)
    attrs = join(string.(attrs), " ")
    res = if isempty(attrs) "<table>" else "<table $attrs>" end
    for row in eachrow(table)
        res *= "\n\t<tr>"
        for x in row
            res *= "\n\t\t<td>$x</td>"
        end
        res *= "\n\t</tr>"
    end
    res *= "\n</table>"
    res
end

function html_render(;open=true, publish::Union{Bool,Nothing}=nothing)
    if isnothing(publish)
        publish = occursin("csail.mit.edu", gethostname())
    end
    full_path = joinpath(Base.Filesystem.pwd(), "out/html", curr_html.dir)
    res = """
    <html>
        <head>
            <h1>$(curr_html.head) @ $(Dates.format(curr_html.timestamp, "yyyy-mm-dd HH:MM:SS"))</h1>
        </head>
        <style>
            $(curr_html.styles)
        </style>

        <script src="https://ajax.googleapis.com/ajax/libs/jquery/1.8.3/jquery.min.js"></script>
        <script>
            $(curr_html.scripts)
        </script>

        <body>
            $(curr_html.body)
        </body>

        <footer>
        <h4>Command to publish</h4>
        <code>julia> using Atari; Atari.html_publish("$full_path")</code>
        </footer>
    </html>
    """
    res = string(res)
    # println(res)
    write("out/html/$(curr_html.dir)/index.html",res)
    println("wrote to out/html/$(curr_html.dir)/index.html")
    open && !publish && open_in_default_browser("out/html/$(curr_html.dir)/index.html")
    publish && html_publish(full_path)
    
end

function html_publish(path)
    @assert isdir(path)
    while endswith(path,"/")
        path = path[:end-1] # we need to trim trailing "/" to make basename() work
    end
    publish_site = get_secret("publish_site") * "/" * basename(path)
    if occursin("csail.mit.edu", gethostname())
        publish_dir = get_secret("publish_dir") * "/" * basename(path)
        println("On CSAIL network: copying files to public site")
        cp(path, publish_dir)
        println("See results at: $publish_site")
    else
        println("Not on CSAIL network: attempting rsync")
        publish_dir = get_secret("publish_dir")
        publish_ssh = get_secret("publish_ssh")
        Base.run(`rsync -avz $path $publish_ssh:$publish_dir/`)
        open_in_default_browser(publish_site)
    end
end

function get_secret(key)
    file = ".secret_$key"
    isfile(file) || error("file $file does not exist. If you'd like to use this feature, add the secret locally but do NOT commit it to git.")

    open(file) do f
        return read(f, String)
    end
end

function detectwsl()
    Sys.islinux() &&
    isfile("/proc/sys/kernel/osrelease") &&
    occursin(r"Microsoft|WSL"i, read("/proc/sys/kernel/osrelease", String))
end

function open_in_default_browser(url::AbstractString)::Bool
    try
        if Sys.isapple()
            Base.run(`open $url`)
            return true
        elseif Sys.iswindows() || detectwsl()
            # Base.run(`cmd.exe /s /c start "" /b $url`)
            url = "file://wsl.localhost/Ubuntu/home/jssteele/probabilistic-atari/$url"
            Base.run(`cmd.exe /s /c start chrome $url`)
            return true
        elseif Sys.islinux()'
            browser = "xdg-open"
            if isfile(browser)
                Base.run(`$browser $url`)
                return true
            else
                @warn "Unable to find `xdg-open`. Try `apt install xdg-open`"
                return false
            end
        else
            return false
        end
    catch ex
        return false
    end
end


const fresh = html_fresh
const render = html_render
