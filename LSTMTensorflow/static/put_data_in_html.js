/**
 * Created by mahsa on 19.06.16.
 */
function put_data(data) {
    if (data != 0 || data != null) {
        alert('success');


        // create an object with the key of the array
        for (var i=0;i<data.length;++i)
            {
                $('#showing_success_tr').append('<td class="name">' + data[i] + ', </td>');

            };
    };
};
function showclassifications_data(showclassifications) {
    // $('#show_classification').html(showclassifications_p);

    $('#show_classification').html(showclassifications.replace(new RegExp('\n', 'g'), '<br/>'));
};

(function () {    
    $('.loading').css('display', 'none');
    return true;
});
